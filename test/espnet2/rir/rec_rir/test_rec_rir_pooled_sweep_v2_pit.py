"""Rec-RIR features, Baseline pooling/head, and unchanged Sweep v2 checks.

CPU tests replace only the Mamba kernels. Actual CUDA Mamba, full-size
training data, and memory use are checked by the experiment GPU preflight.
"""
import io
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn

from espnet2.rir.rec_rir.model_pit import BiSpatialNetPIT
from espnet2.rir.rec_rir.pooled_bimamba import PooledBiMambaCTFPredictor
from espnet2.rir.rec_rir.rec_rir_pooled_sweep_v2_pit import (
    BaselinePooledCTFDecoder,
    ESPnetRecRIRPooledSweepV2PITModel,
    RecRIRPooledCTFPredictor,
)
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)
from espnet2.tasks.rir import RIRTask


SMALL = dict(dim_hidden=8, dim_squeeze=2, num_layers_spch=2,
             num_layers_noise=2, num_layers_CTF=0, encoder_kernel_size=3,
             kernel_size=(3, 3), conv_groups=(2, 2), full_share=0)
DECODER = dict(emb_dim=8, pooling_hidden=4, post_layers=2, n_heads=2,
               ffn_dim=8, local_kernel=3, head_dim=8)
COMMON = dict(sr=8000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
              num_spk=2, ctf_taps=5, pim_sweep_duration=0.1)
REMOVED = ('ctf_layers', 'weight_layer', 'decoder_CTF',
           'decoder_spch', 'decoder_rev')


def cpu_kernel(dim, attention):
    return nn.Sequential(nn.Linear(dim, dim), nn.SiLU())


def patch_mamba():
    context = ExitStack()
    for module in ('model', 'pooled_bimamba'):
        context.enter_context(patch(
            'espnet2.rir.rec_rir.' + module + '._build_mamba', cpu_kernel))
    return context


def make_model():
    with patch_mamba():
        return ESPnetRecRIRPooledSweepV2PITModel(
            **COMMON, network_conf=SMALL, decoder_conf=DECODER)


def make_batch():
    batch = dict(speech_mix=torch.randn(2, 256),
                 speech_mix_lengths=torch.tensor([256, 192]))
    batch['speech_mix'][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = float(speaker)
        reverb = direct.clone()
        reverb[:, 160 + speaker] = speaker * 0.2
        reverb[:, 235 + speaker] = -speaker * 0.1
        batch[f'rir_direct{speaker}'] = direct
        batch[f'rir_ref{speaker}'] = reverb
    return batch


class FixedFeatures(nn.Module):
    """Supply known features through the existing Baseline's real forward."""

    def __init__(self, features):
        super().__init__()
        self.register_buffer('features', features)

    def forward(self, observation):
        return self.features


def test_decoder_matches_actual_baseline_tail_and_padding_policy():
    torch.manual_seed(0)
    decoder = BaselinePooledCTFDecoder(
        input_dim=12, num_spk=2, ctf_taps=5, **DECODER).eval()
    with patch_mamba():
        baseline = PooledBiMambaCTFPredictor(
            input_dim=11, num_spk=2, ctf_taps=5, pre_layers=1,
            freq_bottleneck=2, freq_rank=2, **DECODER).eval()
    for name in ('pooling', 'post_blocks', 'head'):
        getattr(baseline, name).load_state_dict(getattr(decoder, name).state_dict())
    with torch.no_grad():
        baseline.slot_embeddings.copy_(decoder.slot_embeddings)
    baseline.time_blocks = nn.ModuleList([nn.Identity()])
    baseline.frequency_blocks = nn.ModuleList([nn.Identity()])
    # F != T detects accidental time/frequency transposes. The second case
    # appends padding without masking it, exactly as in the existing Baseline.
    features = torch.randn(2, 11, 7, 12)
    for frames in (7, 10):
        value = (features if frames == 7 else
                 torch.cat((features, torch.zeros(2, 11, 3, 12)), dim=2))
        with torch.no_grad():
            projected = decoder.projection(value)
            baseline.encoder = FixedFeatures(projected.permute(0, 3, 2, 1))
            expected = baseline(torch.zeros(2, frames, 11,
                                            dtype=torch.complex64))[0]
            actual = decoder(value)
        assert actual.shape == (2, 2, 11, 5)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rec_trunk_preserves_feature_fusion_axis_and_shared_weights():
    torch.manual_seed(0)
    with patch_mamba():
        original = BiSpatialNetPIT(
            dim_input=2, dim_output_spch=4, dim_output_CTF=20,
            num_freqs=33, num_spk=2, norms=('LN',) * 6, **SMALL).eval()
        predictor = RecRIRPooledCTFPredictor(
            33, 2, 5, network_conf=SMALL, decoder_conf=DECODER).eval()
    feature_state = {name: value for name, value in original.state_dict().items()
                     if not name.startswith(REMOVED)}
    predictor.trunk.load_state_dict(feature_state, strict=True)
    captured = []
    hook = original.compress_CTF.register_forward_hook(
        lambda module, args, output: captured.append(output))
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    ri = torch.stack((observation.real, observation.imag), 1).transpose(-1, -2)
    try:
        with torch.no_grad():
            original(ri.contiguous())
            expected_features = captured[0]
            actual_features = predictor.trunk(ri.contiguous())
            actual_ctf = predictor(observation)[0]
    finally:
        hook.remove()
    assert actual_features.shape == (2, 33, 7, 8)
    torch.testing.assert_close(actual_features, expected_features, rtol=0, atol=0)
    torch.testing.assert_close(actual_ctf, predictor.decoder(expected_features),
                               rtol=0, atol=0)
    for name in REMOVED:
        assert not hasattr(predictor.trunk, name), name
        assert not any(key.startswith(name + '.')
                       for key in predictor.trunk.state_dict()), name
    # full_share=0 intentionally reuses the full-band mixer inside each stage.
    for layers in (predictor.trunk.noise_layers, predictor.trunk.spch_layers):
        assert layers[0].full is layers[1].full


def test_shared_sweep_only_loss_full_tail_gradients_and_teacher_pair_pit():
    torch.manual_seed(0)
    model = make_model()
    batch = make_batch()
    loss, stats, _ = model(**batch)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss.detach(), stats['loss_sweep'], rtol=0, atol=0)
    assert 'loss_speech_rec' not in stats and 'loss_drr' not in stats
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for component in ('trunk.encoder', 'trunk.noise_layers', 'trunk.spch_layers',
                      'trunk.compress_CTF', 'decoder.projection', 'decoder.pooling',
                      'decoder.slot_embeddings', 'decoder.post_blocks', 'decoder.head'):
        values = [parameter.grad.abs().sum()
                  for name, parameter in model.named_parameters()
                  if name.startswith('ctf_predictor.' + component)
                  and parameter.requires_grad]
        assert values and sum(values) > 0, component
    model.eval()
    swapped = {name: value.clone() for name, value in batch.items()}
    for prefix in ('rir_direct', 'rir_ref'):
        swapped[prefix + '1'][0] = batch[prefix + '2'][0]
        swapped[prefix + '2'][0] = batch[prefix + '1'][0]
    with torch.no_grad():
        torch.testing.assert_close(model(**swapped)[0], model(**batch)[0])
        speech = batch['speech_mix'] / batch['speech_mix'].abs().amax(1, keepdim=True)
        ctf = model._estimate_ctf_from_complex(model.transforms.stft(speech[:, None]))
        direct = torch.stack([batch[f'rir_direct{s}'] for s in (1, 2)], 1)
        reverb = torch.stack([batch[f'rir_ref{s}'] for s in (1, 2)], 1)
        shared = ESPnetTFLocoformerSweepV2PITModel(
            **COMMON, ctf_predictor=nn.Identity())
        expected, permutation = shared.paired_pit_loss(ctf, direct, reverb)
        actual, actual_permutation = model.paired_pit_loss(ctf, direct, reverb)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_permutation, permutation)
        torch.testing.assert_close(model(**batch)[0].reshape(()), expected)
    # An error restricted to the delayed response must reach the final CTF tap.
    direct = torch.zeros(1, 2, 256)
    direct[:, 0, 20] = 1
    direct[:, 1, 30] = -2
    delayed = torch.zeros_like(direct)
    delayed[:, 0, 160] = 0.5
    delayed[:, 1, 170] = -0.75
    wrong = torch.zeros(1, 2, 33, 5, dtype=torch.complex64, requires_grad=True)
    tail_loss, _ = model.paired_pit_loss(wrong, direct, delayed)
    tail_loss.backward()
    assert tail_loss > 0 and wrong.grad[..., -1].abs().sum() > 0


def test_causal_tap_order_checkpoint_and_pim_roundtrip():
    torch.manual_seed(1)
    model = make_model().eval()
    batch = make_batch()
    with torch.no_grad():
        speech = batch['speech_mix'] / batch['speech_mix'].abs().amax(1, keepdim=True)
        causal = model._estimate_ctf_from_complex(model.transforms.stft(speech[:, None]))
        # The shared PIM interface reverses causal taps exactly once.
        torch.testing.assert_close(model.estimate_ctf(batch['speech_mix'][0]),
                                   causal[0].flip(-1))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = make_model().eval()
        clone.load_state_dict(torch.load(buffer), strict=True)
        expected = model.estimate_rir(batch['speech_mix'][0], rir_length=256)
        actual = clone.estimate_rir(batch['speech_mix'][0], rir_length=256)
        assert actual.shape == (2, 256) and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_recipe_matches_baseline_conditions_and_sweep_only_registration():
    root = Path(__file__).resolve().parents[4]
    configs = root / 'egs2/whamr/rir_2spk/conf/tuning'
    baseline = yaml.safe_load((
        configs / 'train_pooled_bimamba_2spk_nf_8k_sweep_v2.yaml').read_text())
    path = configs / 'train_rec_rir_pooled_2spk_nf_8k_sweep_v2.yaml'
    new = yaml.safe_load(path.read_text())
    expected = dict(baseline)
    expected['rir_model_type'] = 'rec_rir_pooled_sweep_v2_pit'
    expected['model_conf'] = dict(expected['model_conf'])
    del expected['model_conf']['predictor_conf']
    expected['model_conf'].update(network_conf=new['model_conf']['network_conf'],
                                  decoder_conf=new['model_conf']['decoder_conf'])
    expected['model_conf']['drr_loss_weight'] = 0.0
    assert new == expected
    assert new['model_conf']['network_conf']['num_layers_CTF'] == 0
    assert new['model_conf'].get('drr_loss_weight', 0) == 0
    assert new['model_conf']['decoder_conf']['post_layers'] == 2
    assert new['model_conf']['decoder_conf']['emb_dim'] == 64
    args = RIRTask.get_parser().parse_args([
        '--config', str(path), '--output_dir', 'unused'])
    args.model_conf['network_conf'] = SMALL
    args.model_conf['decoder_conf'] = DECODER
    with patch_mamba():
        model = RIRTask.build_model(args)
    assert isinstance(model, ESPnetRecRIRPooledSweepV2PITModel)
    assert model.paired_pit_loss.__func__ is ESPnetTFLocoformerSweepV2PITModel.paired_pit_loss
    assert model.drr_loss_weight == 0
    assert model.ctf_predictor.decoder.ctf_taps == 60
    assert type(RIRTask.build_preprocess_fn(args, True)).__name__ == 'RecRIRSweepV2PITPreprocessor'


def test_rejects_ctf_temporal_layers_and_auxiliary_losses():
    with patch_mamba():
        with unittest.TestCase().assertRaises(ValueError):
            RecRIRPooledCTFPredictor(
                33, 2, 5, network_conf=dict(SMALL, num_layers_CTF=6),
                decoder_conf=DECODER)
        with unittest.TestCase().assertRaises(ValueError):
            ESPnetRecRIRPooledSweepV2PITModel(
                **COMMON, network_conf=SMALL, decoder_conf=DECODER,
                drr_loss_weight=1.0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in list(globals().items()) if name.startswith('test_'))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
