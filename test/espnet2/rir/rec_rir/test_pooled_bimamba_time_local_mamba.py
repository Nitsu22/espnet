"""Exact window/reset tests; CUDA validation also compares official Mamba."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from espnet2.rir.rec_rir.pooled_bimamba import PooledBiMambaCTFPredictor, TimeLocalBiMamba
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import ESPnetPooledBiMambaSweepV2PITModel


def reference_scan(u, delta, a, b, c, d, z, delta_bias, delta_softplus):
    """Independent CPU state recurrence, including gate and skip term."""
    steps = F.softplus(delta + delta_bias[None, :, None])
    state = u.new_zeros(u.shape[0], u.shape[1], a.shape[1])
    outputs = []
    for t in range(u.shape[-1]):
        dt = steps[:, :, t, None]
        state = torch.exp(dt * a[None]) * state + dt * b[:, None, :, t] * u[:, :, t, None]
        value = (state * c[:, None, :, t]).sum(-1) + d[None] * u[:, :, t]
        outputs.append(value * F.silu(z[:, :, t]))
    return torch.stack(outputs, -1)


class CpuMambaOracle(nn.Module):
    """Mamba-1 equations with explicit causal convolution and CPU recurrence."""
    def __init__(self, dim, d_state, d_conv):
        super().__init__()
        self.d_inner, self.d_state, self.d_conv, self.dt_rank = 2 * dim, d_state, d_conv, 1
        self.in_proj = nn.Linear(dim, 4 * dim, bias=False)
        self.conv1d = nn.Conv1d(2 * dim, 2 * dim, d_conv, groups=2 * dim)
        self.x_proj = nn.Linear(2 * dim, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, 2 * dim)
        self.A_log = nn.Parameter(torch.randn(2 * dim, d_state))
        self.D = nn.Parameter(torch.ones(2 * dim))
        self.out_proj = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, sequence):
        x, z = self.in_proj(sequence).chunk(2, -1)
        values = []
        for t in range(sequence.shape[1]):
            value = self.conv1d.bias[None].expand(sequence.shape[0], -1)
            for lag in range(self.d_conv):
                if t >= lag:
                    value = value + x[:, t-lag] * self.conv1d.weight[:, 0, -1-lag]
            values.append(F.silu(value))
        u = torch.stack(values, 1)
        delta, b, c = self.x_proj(u).split([1, self.d_state, self.d_state], -1)
        delta = F.linear(delta, self.dt_proj.weight).transpose(1, 2)
        y = reference_scan(u.transpose(1, 2), delta, -self.A_log.float().exp(),
                           b.transpose(1, 2), c.transpose(1, 2), self.D.float(),
                           z.transpose(1, 2), self.dt_proj.bias.float(), True)
        return self.out_proj(y.transpose(1, 2))


def cpu_mamba(dim, specification):
    state, conv = map(int, specification[6:-1].split(','))
    return CpuMambaOracle(dim, state, conv)


def per_frame_oracle(block, x):
    batch, frames, freqs, dim = x.shape
    sequence = F.layer_norm(x, (dim,), block.norm.weight, block.norm.bias, block.norm.eps)
    sequence = sequence.permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
    results = []
    for t in range(frames):
        first, last = max(0, t-block.radius), min(frames, t+block.radius+1)
        window = sequence[:, first:last]
        forward = block.forward_mamba(window)[:, t-first]
        backward = block.backward_mamba(window.flip(1)).flip(1)[:, t-first]
        results.append(block.projection(torch.cat((forward, backward), -1)))
    update = torch.stack(results, 1).reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)
    return x + update


def test_clipped_windows_match_independent_oracle_output_and_all_gradients():
    for frames, window, chunk in ((7, 1, 3), (11, 5, 7), (9, 125, 4), (133, 125, 257)):
        with patch('espnet2.rir.rec_rir.pooled_bimamba._build_mamba', cpu_mamba):
            torch.manual_seed(3)
            block = TimeLocalBiMamba(4, 2, 4, window, 1e-5, chunk)
        oracle = copy.deepcopy(block)
        x = torch.randn(1, frames, 2, 4, requires_grad=True)
        ref = x.detach().clone().requires_grad_()
        with patch.object(TimeLocalBiMamba, '_selective_scan', staticmethod(reference_scan)):
            actual = block(x)
            expected = per_frame_oracle(oracle, ref)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
            probe = torch.randn_like(actual)
            (actual * probe).sum().backward()
            (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, ref.grad, rtol=5e-5, atol=1e-5)
        refs = dict(oracle.named_parameters())
        for name, param in block.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            torch.testing.assert_close(param.grad, refs[name].grad, rtol=1e-4, atol=5e-5)


def test_no_outside_window_batch_or_frequency_dependency():
    with patch('espnet2.rir.rec_rir.pooled_bimamba._build_mamba', cpu_mamba):
        block = TimeLocalBiMamba(4, 2, 3, 5, 1e-5, 7).eval()
    x = torch.randn(2, 11, 3, 4, requires_grad=True)
    with patch.object(TimeLocalBiMamba, '_selective_scan', staticmethod(reference_scan)):
        y = block(x)
        for b, t, f in ((0, 2, 1), (0, 8, 1), (1, 5, 1), (0, 5, 2)):
            modified = x.detach().clone()
            modified[b, t, f, 0] += 7
            torch.testing.assert_close(block(modified)[0, 5, 1], y[0, 5, 1], rtol=0, atol=0)
        y[0, 5, 1].square().sum().backward()
    gradients = x.grad.abs().sum(-1)
    assert gradients[0, 3, 1] > 0 and gradients[0, 7, 1] > 0
    gradients[0, 3:8, 1] = 0
    assert gradients.count_nonzero() == 0


SMALL = dict(emb_dim=8, pre_layers=2, post_layers=1, d_state=2, d_conv=3,
             freq_bottleneck=4, freq_rank=4, local_kernel=3, pooling_hidden=4,
             n_heads=2, ffn_dim=8, head_dim=8)


def test_default_checkpoint_shared_weights_and_rng_and_stage_order():
    with patch('espnet2.rir.rec_rir.pooled_bimamba._build_mamba', cpu_mamba):
        models, rng = [], []
        for options in ({}, {'time_local_bimamba_window': 0}, {'time_local_bimamba_window': 5}):
            torch.manual_seed(7)
            models.append(PooledBiMambaCTFPredictor(input_dim=9, ctf_taps=3, **SMALL, **options))
            rng.append(torch.get_rng_state())
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    for model in models[1:]:
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for value in rng[1:]:
        torch.testing.assert_close(value, rng[0], rtol=0, atol=0)
    events, handles = [], []
    for i in range(2):
        for label, layer in [('local', models[2].time_local_bimamba_blocks[i]),
                             ('time', models[2].time_blocks[i]), ('frequency', models[2].frequency_blocks[i])]:
            handles.append(layer.register_forward_pre_hook(lambda m, x, i=i, l=label: events.append((i, l))))
    with patch.object(TimeLocalBiMamba, '_selective_scan', staticmethod(reference_scan)):
        spectrum = torch.randn(2, 7, 9, dtype=torch.complex64)
        torch.testing.assert_close(models[0](spectrum)[0], models[1](spectrum)[0], rtol=0, atol=0)
        output = models[2](spectrum)[0]
    for handle in handles:
        handle.remove()
    assert output.shape == (2, 2, 9, 3) and torch.is_complex(output)
    assert events == [(i, l) for i in range(2) for l in ('local', 'time', 'frequency')]


def test_config_only_changes_local_module_and_memory_execution_options():
    configs = Path(__file__).resolve().parents[4] / 'egs2/whamr/rir_2spk/conf/tuning'
    prefix = 'train_pooled_bimamba_2spk_nf_16k_sweep_v2_time_local_'
    attention = yaml.safe_load((configs / (prefix+'attention125.yaml')).read_text())
    expected = copy.deepcopy(attention)
    conf = expected['model_conf']['predictor_conf']
    del conf['time_local_attention_window']
    conf.update(time_local_bimamba_window=125, time_local_bimamba_window_batch_size=2048,
                time_local_bimamba_checkpoint=True)
    assert yaml.safe_load((configs / (prefix+'bimamba125.yaml')).read_text()) == expected


def test_window_batches_are_bounded_and_checkpoint_does_not_change_gradients():
    with patch('espnet2.rir.rec_rir.pooled_bimamba._build_mamba', cpu_mamba):
        block = TimeLocalBiMamba(4, 2, 3, 5, 1e-5, 7)
    reference = copy.deepcopy(block)
    reference.window_batch_size = 1000
    reference.activation_checkpoint = False
    seen = []
    def observe(*args, **kwargs):
        seen.append(args[0].shape)
        return reference_scan(*args, **kwargs)
    x = torch.randn(2, 11, 3, 4, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    with patch.object(TimeLocalBiMamba, '_selective_scan', staticmethod(observe)):
        actual = block(x)
        actual.square().sum().backward()
    assert seen and all(shape[0] <= 7 and shape[-1] == 3 for shape in seen)
    with patch.object(TimeLocalBiMamba, '_selective_scan', staticmethod(reference_scan)):
        expected = reference(ref)
        expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, ref.grad)
    for a, b in zip(block.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=5e-5)


def test_invalid_local_configuration_fails_before_building_kernels():
    for window in (0, -1, 2, True, 1.0):
        with unittest.TestCase().assertRaises(ValueError):
            TimeLocalBiMamba(4, 2, 3, window, 1e-5)
    for value in (0, -1, True, 1.0):
        with unittest.TestCase().assertRaises(ValueError):
            TimeLocalBiMamba(4, 2, 3, 5, 1e-5, value)
    for options in ({'time_local_bimamba_window': -1},
                    {'time_local_bimamba_window': 2},
                    {'time_local_bimamba_window': True},
                    {'time_local_bimamba_window_batch_size': 0},
                    {'time_local_bimamba_checkpoint': 'true'},
                    {'time_local_attention_window': 5, 'time_local_bimamba_window': 5},
                    {'time_block_indices': [], 'time_local_bimamba_window': 5}):
        with unittest.TestCase().assertRaises(ValueError):
            PooledBiMambaCTFPredictor(input_dim=9, ctf_taps=3, **SMALL, **options)


def test_shared_sweep_loss_pit_gradients_and_checkpoint_two_rirs():
    options = dict(time_local_bimamba_window=5, time_local_bimamba_window_batch_size=7)
    def model(local=True):
        return ESPnetPooledBiMambaSweepV2PITModel(
            sr=16000, n_fft=32, win_len=32, hop_len=16, num_freqs=17,
            ctf_taps=3, pim_sweep_duration=0.1, predictor_conf=SMALL | (options if local else {}))
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_lengths=torch.tensor([256, 192]))
    batch['speech_mix'][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20+speaker] = speaker
        reverb = direct.clone()
        reverb[:, 180+speaker] = .3 * speaker
        batch['rir_direct'+str(speaker)], batch['rir_ref'+str(speaker)] = direct, reverb
    with patch('espnet2.rir.rec_rir.pooled_bimamba._build_mamba', cpu_mamba), \
            patch.object(TimeLocalBiMamba, '_selective_scan', staticmethod(reference_scan)):
        actual, baseline = model(), model(False).eval()
        loss, stats, _ = actual(**batch)
        torch.testing.assert_close(loss.detach(), stats['loss_sweep'])
        loss.backward()
        for name, p in actual.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
        actual.eval()
        swapped = dict(batch)
        for prefix in ('rir_direct', 'rir_ref'):
            swapped[prefix+'1'], swapped[prefix+'2'] = batch[prefix+'2'], batch[prefix+'1']
        with torch.no_grad():
            torch.testing.assert_close(actual(**batch)[0], actual(**swapped)[0])
            spectrum = actual.transforms.stft(batch['speech_mix'][:, None], 'complex')
            ctf, _ = actual._estimate_ctf_and_aux_from_complex(spectrum)
            direct = torch.stack([batch['rir_direct'+str(s)] for s in (1, 2)], 1)
            reverb = torch.stack([batch['rir_ref'+str(s)] for s in (1, 2)], 1)
            for a, b in zip(actual.paired_pit_loss(ctf, direct, reverb), baseline.paired_pit_loss(ctf, direct, reverb)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            clone = model().eval()
            clone.load_state_dict(actual.state_dict(), strict=True)
            torch.testing.assert_close(actual.estimate_ctf(batch['speech_mix'][0]), clone.estimate_ctf(batch['speech_mix'][0]))
            rir = clone.estimate_rir(batch['speech_mix'][0], rir_length=256)
            assert rir.shape == (2, 256) and torch.isfinite(rir).all()


if __name__ == '__main__':
    tests = [unittest.FunctionTestCase(value) for name, value in list(globals().items()) if name.startswith('test_')]
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests)).wasSuccessful())
