"""Rec-RIR path parity and shared Sweep v2 PIT/PIM regression tests.

CPU tests replace only Mamba kernels. Real Mamba is checked on the lab GPU.
"""
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn

from espnet2.rir.rec_rir.model_pit import BiSpatialNetPIT
from espnet2.rir.rec_rir.rec_rir_sweep_v2_pit import (
    ESPnetRecRIRSweepV2PITModel, RecRIRCTFPredictor,
)
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)
from espnet2.tasks.rir import RIRTask


SMALL = dict(dim_hidden=8, dim_squeeze=2, num_layers_spch=1,
             num_layers_noise=1, num_layers_CTF=1, encoder_kernel_size=3,
             kernel_size=(3, 3), conv_groups=(2, 2), full_share=0)
COMMON = dict(sr=8000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
              num_spk=2, ctf_taps=5, pim_sweep_duration=0.1)


def cpu_kernel(dim, attention):
    return nn.Sequential(nn.Linear(dim, dim), nn.SiLU())


def make_model():
    with patch('espnet2.rir.rec_rir.model._build_mamba', cpu_kernel):
        return ESPnetRecRIRSweepV2PITModel(**COMMON, network_conf=SMALL)


def test_existing_rec_rir_ctf_path_and_ri_speaker_tap_layout_are_preserved():
    with patch('espnet2.rir.rec_rir.model._build_mamba', cpu_kernel):
        old = BiSpatialNetPIT(dim_input=2, dim_output_spch=4,
            dim_output_CTF=20, num_freqs=33, num_spk=2, norms=('LN',) * 6,
            **SMALL).eval()
        new = RecRIRCTFPredictor(33, 2, 5, SMALL).eval()
    state = {key: value for key, value in old.state_dict().items()
             if not key.startswith(('decoder_spch.', 'decoder_rev.'))}
    new.network.load_state_dict(state, strict=True)
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    ri = torch.stack((observation.real, observation.imag), 1).transpose(-1, -2)
    with torch.no_grad():
        features = old(ri.contiguous())[1]
        expected = torch.complex(features[:, :, 0], features[:, :, 1])
        actual = new(observation)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 2, 33, 5)
    assert not any('decoder_spch' in key or 'decoder_rev' in key
                   for key in new.state_dict())


def test_paired_pit_shared_loss_gradients_and_checkpoint_pim_roundtrip():
    torch.manual_seed(0)
    model = make_model()
    batch = dict(speech_mix=torch.randn(2, 256),
                 speech_mix_lengths=torch.tensor([256, 192]))
    batch['speech_mix'][1, 192:] = 0
    for s in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + s] = float(s)
        reverb = direct.clone()
        reverb[:, 160 + s] = s * 0.2
        batch[f'rir_direct{s}'] = direct
        batch[f'rir_ref{s}'] = reverb
    loss, stats, _ = model(**batch)
    torch.testing.assert_close(loss.detach(), stats['loss_sweep'])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    assert all(p.grad.abs().sum() > 0 for key, p in model.named_parameters()
               if 'decoder_CTF' in key)
    model.eval()
    swapped = {key: value.clone() for key, value in batch.items()}
    for prefix in ('rir_direct', 'rir_ref'):
        swapped[prefix + '1'][0] = batch[prefix + '2'][0]
        swapped[prefix + '2'][0] = batch[prefix + '1'][0]
    with torch.no_grad():
        torch.testing.assert_close(model(**swapped)[0], model(**batch)[0])
        speech = batch['speech_mix'] / batch['speech_mix'].abs().amax(1, keepdim=True)
        ctf = model._estimate_ctf_from_complex(model.transforms.stft(speech[:, None]))
        direct = torch.stack([batch[f'rir_direct{s}'] for s in (1, 2)], 1)
        reverb = torch.stack([batch[f'rir_ref{s}'] for s in (1, 2)], 1)
        baseline = ESPnetTFLocoformerSweepV2PITModel(**COMMON, ctf_predictor=nn.Identity())
        torch.testing.assert_close(model(**batch)[0].reshape(()), baseline.paired_pit_loss(ctf, direct, reverb)[0])
        # Shared inference flips causal taps exactly once for ordinary PIM.
        torch.testing.assert_close(model.estimate_ctf(batch['speech_mix'][0]), ctf[0].flip(-1))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = make_model().eval()
        clone.load_state_dict(torch.load(buffer), strict=True)
        rirs = clone.estimate_rir(batch['speech_mix'][0], rir_length=256)
        assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()
        torch.testing.assert_close(rirs, model.estimate_rir(batch['speech_mix'][0], rir_length=256))


def test_recipe_matches_bimamba_conditions_and_task_registration():
    root = Path(__file__).resolve().parents[4]
    recipe = root / 'egs2/whamr/rir_2spk'
    baseline = yaml.safe_load((recipe / 'conf/tuning/train_pooled_bimamba_2spk_nf_8k_sweep_v2.yaml').read_text())
    path = recipe / 'conf/tuning/train_rec_rir_2spk_nf_8k_sweep_v2.yaml'
    new = yaml.safe_load(path.read_text())
    expected = dict(baseline)
    expected['rir_model_type'] = 'rec_rir_sweep_v2_pit'
    expected['model_conf'] = dict(expected['model_conf'])
    del expected['model_conf']['predictor_conf']
    expected['model_conf']['network_conf'] = new['model_conf']['network_conf']
    assert new == expected
    args = RIRTask.get_parser().parse_args(['--config', str(path), '--output_dir', 'unused'])
    args.model_conf['network_conf'] = SMALL
    with patch('espnet2.rir.rec_rir.model._build_mamba', cpu_kernel):
        model = RIRTask.build_model(args)
    assert isinstance(model, ESPnetRecRIRSweepV2PITModel)
    assert model.paired_pit_loss.__func__ is ESPnetTFLocoformerSweepV2PITModel.paired_pit_loss
    assert model.drr_loss_weight == 0
    assert model.ctf_predictor.network.ctf_taps == 60
    assert type(RIRTask.build_preprocess_fn(args, True)).__name__ == 'RecRIRSweepV2PITPreprocessor'


if __name__ == '__main__':
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for key, value in list(globals().items()) if key.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
