"""Standalone CPU objective checks and optional real-Mamba CUDA checks."""
import io
import unittest
from pathlib import Path

import torch
import yaml

from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import ESPnetPooledBiMambaSweepV2PITModel
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import ESPnetTFLocoformerSweepV2PITModel


def small_model():
    return ESPnetPooledBiMambaSweepV2PITModel(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, pim_sweep_duration=.1,
        predictor_conf=dict(emb_dim=8, pre_layers=1, post_layers=1,
                            d_state=4, d_conv=2, freq_bottleneck=4, freq_rank=4,
                            pooling_hidden=4, n_heads=2, ffn_dim=8, head_dim=8))


def test_identical_objective_and_full_tail():
    new = small_model()
    baseline = ESPnetTFLocoformerSweepV2PITModel(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, pim_sweep_duration=.1, n_layers=1, emb_dim=8,
        num_groups=2, n_heads=2, attention_dim=8, pos_enc='nope',
        ffn_hidden_dim=[8, 8], conv1d_kernel=3)
    direct = torch.zeros(2, 2, 256)
    direct[:, 0, 20] = 1
    direct[:, 1, 30] = -2
    target = torch.zeros_like(direct)
    target[:, 0, 20] = 3
    target[:, 1, 158] = -1
    ctf = torch.zeros(2, 2, 33, 5, dtype=torch.complex64)
    ctf[:, 0, :, 0] = 3
    ctf[:, 1, :, 4] = .5
    a, permutation_a = new.paired_pit_loss(ctf, direct, target)
    b, permutation_b = baseline.paired_pit_loss(ctf, direct, target)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(permutation_a, permutation_b)
    torch.testing.assert_close(a, new.paired_pit_loss(ctf.flip(1), direct, target)[0])
    wrong = torch.zeros_like(ctf, requires_grad=True)
    bad, _ = new.paired_pit_loss(wrong, direct, target)
    bad.backward()
    assert bad > 0 and wrong.grad[..., 4].abs().sum() > 0


def test_shared_training_conditions():
    root = Path(__file__).resolve().parents[4]
    configs = root / 'egs2/whamr/rir_2spk/conf/tuning'
    old = yaml.safe_load((configs / 'train_tflocoformer_2spk_nf_16k_sweep_v2.yaml').read_text())
    new = yaml.safe_load((configs / 'train_pooled_bimamba_2spk_nf_16k_sweep_v2.yaml').read_text())
    for key in old.keys() - {'model_conf', 'rir_model_type'}:
        assert old[key] == new[key], key
    for key, value in new['model_conf'].items():
        if key != 'predictor_conf':
            assert old['model_conf'][key] == value, key


def test_cuda_predictor_gradients_checkpoint_and_inference():
    if not torch.cuda.is_available():
        raise unittest.SkipTest('Real Mamba forward requires CUDA')
    torch.manual_seed(0)
    model = small_model().cuda().train()
    batch = dict(speech_mix=torch.randn(2, 256, device='cuda'),
                 speech_mix_lengths=torch.tensor([256, 192], device='cuda'))
    # Preserve baseline unmasked behavior for a shorter padded observation.
    batch['speech_mix'][1, 192:] = 0
    for index in (1, 2):
        direct = torch.zeros(2, 256, device='cuda')
        direct[:, 20 + index] = index
        reverb = direct.clone()
        reverb[:, 100 + index] = .25 * index
        batch[f'rir_direct{index}'] = direct
        batch[f'rir_ref{index}'] = reverb
    loss, _, _ = model(**batch)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for component in ('encoder', 'time_blocks', 'frequency_blocks', 'pooling',
                      'slot_embeddings', 'post_blocks', 'head'):
        gradients = [p.grad.abs().sum() for n, p in model.named_parameters()
                     if p.requires_grad and n.startswith('ctf_predictor.' + component)]
        assert sum(gradients) > 0, component
    model.eval()
    with torch.no_grad():
        ctf = model.estimate_ctf(batch['speech_mix'][0])
        assert ctf.shape == (2, 33, 5)
        assert torch.is_complex(ctf) and torch.isfinite(ctf).all()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = small_model().cuda().eval()
        clone.load_state_dict(torch.load(buffer), strict=True)
        torch.testing.assert_close(ctf, clone.estimate_ctf(batch['speech_mix'][0]))
        rirs = clone.estimate_rir(batch['speech_mix'][0], rir_length=256)
        assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for key, value in list(globals().items()) if key.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
