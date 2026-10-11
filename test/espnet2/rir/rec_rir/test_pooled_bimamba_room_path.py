"""Standalone room/path structure checks with real CPU BiLSTM integration.

The Mamba substitute is used only for topology and seeded initialization;
these tests do not exercise CUDA Mamba kernels or establish disentanglement
of physical room and source/receiver path properties.
"""

import copy
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from espnet2.rir.rec_rir.pooled_bimamba import PooledBiMambaCTFPredictor
from espnet2.rir.rec_rir.pooled_bimamba_room_path import (
    RoomPathPooledBiMambaCTFPredictor,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)
from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)


TINY = dict(
    emb_dim=8,
    pre_layers=1,
    post_layers=2,
    d_state=4,
    d_conv=2,
    freq_bottleneck=4,
    freq_rank=4,
    local_kernel=3,
    pooling_hidden=4,
    n_heads=2,
    ffn_dim=8,
    head_dim=8,
)


def _predictor(**options):
    return RoomPathPooledBiMambaCTFPredictor(
        input_dim=33, ctf_taps=5, **(TINY | {"time_module": "bilstm"} | options)
    )


def _sweep_model(**options):
    return ESPnetPooledBiMambaSweepV2PITModel(
        sr=16000,
        n_fft=64,
        win_len=64,
        hop_len=32,
        num_freqs=33,
        ctf_taps=5,
        pim_sweep_duration=0.1,
        predictor_conf=TINY | {"time_module": "bilstm"} | options,
    )


def _cpu_mamba_substitute(dim, specification):
    """Only a trainable topology/RNG substitute; integration uses real LSTM."""
    return nn.Sequential(nn.Linear(dim, dim), nn.SiLU())


def _temporal_weights(scores):
    # Explicit normalized exponentials avoid reusing the tested softmax call.
    unnormalized = (scores - scores.amax(1, keepdim=True)).exp()
    return unnormalized / unnormalized.sum(1, keepdim=True)


def test_three_codes_match_independent_weighted_pooling_reference():
    torch.manual_seed(13)
    model = _predictor().eval()
    captured = []
    handle = model.pooling[0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    observation[1, 5:] = 0
    codes = model.encode_room_path(observation, torch.tensor([7, 5]))
    handle.remove()
    assert list(codes) == ["room", "path1", "path2"]
    assert len(captured) == 1
    features = captured[0]
    hidden = F.gelu(F.linear(
        features, model.pooling[0].weight, model.pooling[0].bias
    ))
    room_scores = F.linear(hidden, model.room_pooling.weight, model.room_pooling.bias)
    room = (_temporal_weights(room_scores) * features).sum(1)
    path_scores = F.linear(hidden, model.pooling[2].weight, model.pooling[2].bias)
    path_scores = path_scores.reshape(2, 7, 33, 2, 8)
    paths = torch.einsum("btfsd,btfd->bsfd", _temporal_weights(path_scores), features)
    paths = paths + model.slot_embeddings[None, :, None]
    torch.testing.assert_close(codes["room"], room)
    torch.testing.assert_close(codes["path1"], paths[:, 0])
    torch.testing.assert_close(codes["path2"], paths[:, 1])
    assert all(value.shape == (2, 33, 8) for value in codes.values())
    assert not torch.allclose(codes["path1"], codes["path2"])
    assert not torch.allclose(codes["room"], features.mean(1))
    # Preserve the comparison baseline's full-axis, unmasked time pooling.
    alternate_lengths = model.encode_room_path(observation, torch.tensor([7, 7]))
    for name in codes:
        torch.testing.assert_close(codes[name], alternate_lengths[name], rtol=0, atol=0)


def test_shared_decoder_uses_exact_same_room_for_both_paths():
    torch.manual_seed(17)
    model = _predictor().eval()
    room = torch.randn(2, 33, 8)
    paths = torch.randn(2, 2, 33, 8)
    captured = []
    handle = model.room_path_fusion.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    combined = model.decode_room_path(room, paths)
    handle.remove()
    assert len(captured) == 1
    fusion_input = captured[0].reshape(2, 2, 33, 16)
    torch.testing.assert_close(fusion_input[..., :8], room[:, None].expand(-1, 2, -1, -1),
                               atol=0, rtol=0)
    torch.testing.assert_close(fusion_input[..., 8:], paths, atol=0, rtol=0)
    individual = torch.stack([
        model.decode_room_path(room, paths[:, slot]) for slot in (0, 1)
    ], dim=1)
    torch.testing.assert_close(combined, individual, atol=2e-6, rtol=2e-6)
    assert combined.shape == (2, 2, 33, 5)
    assert torch.is_complex(combined) and torch.isfinite(combined).all()
    assert len(model.post_interaction_blocks) == 0


def test_decoder_path_independence_shared_room_and_slot_permutation():
    torch.manual_seed(19)
    model = _predictor().eval()
    room = torch.randn(2, 33, 8)
    paths = torch.randn(2, 2, 33, 8)
    baseline = model.decode_room_path(room, paths)
    changed_paths = paths.clone()
    changed_paths[:, 0, :, 0] += 3
    modified = model.decode_room_path(room, changed_paths)
    torch.testing.assert_close(modified[:, 1], baseline[:, 1], atol=0, rtol=0)
    assert not torch.allclose(modified[:, 0], baseline[:, 0])
    changed_room = room.clone()
    changed_room[..., 0] += 3
    room_modified = model.decode_room_path(changed_room, paths)
    assert all(not torch.allclose(room_modified[:, slot], baseline[:, slot])
               for slot in (0, 1))
    torch.testing.assert_close(model.decode_room_path(room, paths.flip(1)),
                               baseline.flip(1), atol=2e-6, rtol=2e-6)
    observation = torch.randn(2, 7, 33, dtype=torch.complex64)
    ctf, lengths, aux = model(observation, torch.tensor([7, 5]))
    assert list(aux) == ["room_code", "path_codes"]
    assert aux["room_code"].shape == (2, 33, 8)
    assert aux["path_codes"].shape == (2, 2, 33, 8)
    torch.testing.assert_close(ctf, model.decode_room_path(aux["room_code"], aux["path_codes"]))
    torch.testing.assert_close(lengths, torch.tensor([7, 5]))


def test_new_modules_preserve_seeded_common_baseline_parameters():
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba", _cpu_mamba_substitute):
        torch.manual_seed(23)
        baseline = PooledBiMambaCTFPredictor(input_dim=33, ctf_taps=5, **TINY)
        torch.manual_seed(23)
        factorized = _predictor(time_module="bimamba")
    old, new = baseline.state_dict(), factorized.state_dict()
    for name, value in old.items():
        torch.testing.assert_close(value, new[name], rtol=0, atol=0)
    additions = new.keys() - old.keys()
    assert additions and all(name.startswith(("room_pooling.", "room_path_fusion."))
                             for name in additions)
    delta = sum(p.numel() for p in factorized.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    )
    # Room score H->D, fusion normalization 2D, projection 2D->D.
    assert delta == 4 * 8 + 8 + 4 * 8 + 2 * 8 * 8 + 8


def test_wrapper_flag_dispatch_default_checkpoint_and_output_unchanged():
    torch.manual_seed(29)
    default = _sweep_model().eval()
    torch.manual_seed(29)
    explicit_false = _sweep_model(room_path_factorized=False).eval()
    assert type(default.ctf_predictor) is PooledBiMambaCTFPredictor
    explicit_false.load_state_dict(default.state_dict(), strict=True)
    waveform = torch.randn(256)
    torch.testing.assert_close(default.estimate_ctf(waveform), explicit_false.estimate_ctf(waveform),
                               rtol=0, atol=0)
    assert default.state_dict().keys() == explicit_false.state_dict().keys()
    factorized = _sweep_model(room_path_factorized=True)
    assert type(factorized.ctf_predictor) is RoomPathPooledBiMambaCTFPredictor
    for flag in ("true", 1, 0, None, []):
        with unittest.TestCase().assertRaisesRegex(ValueError, "room_path_factorized"):
            _sweep_model(room_path_factorized=flag)


def test_real_bilstm_full_sweep_pit_gradients_checkpoint_and_two_rir_inference():
    torch.manual_seed(31)
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba",
               side_effect=AssertionError("CPU BiLSTM requested Mamba")):
        model = _sweep_model(room_path_factorized=True).train()
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_lengths=torch.tensor([256, 192]))
    batch["speech_mix"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = speaker
        reverb = direct.clone()
        reverb[:, 155 + speaker] = 0.3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    loss, stats, count = model(**batch)
    assert torch.isfinite(loss) and count == 2
    torch.testing.assert_close(stats["loss"], stats["loss_sweep"], rtol=0, atol=0)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for component in ("encoder", "time_blocks", "frequency_blocks", "pooling",
                      "slot_embeddings", "room_pooling", "room_path_fusion", "post_blocks", "head"):
        gradients = [p.grad.abs().sum() for name, p in model.named_parameters()
                     if p.requires_grad and name.startswith("ctf_predictor." + component)]
        assert gradients and sum(gradients) > 0, component
    model.eval()
    reference = ESPnetTFLocoformerSweepV2PITModel(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, pim_sweep_duration=0.1, n_layers=1, emb_dim=8,
        num_groups=2, n_heads=2, attention_dim=8, pos_enc="nope",
        ffn_hidden_dim=[8, 8], conv1d_kernel=3,
    )
    direct = torch.stack([batch[f"rir_direct{s}"] for s in (1, 2)], 1)
    reverb = torch.stack([batch[f"rir_ref{s}"] for s in (1, 2)], 1)
    with torch.no_grad():
        spectrum = model.transforms.stft(batch["speech_mix"][:, None], "complex")
        ctf, _ = model._estimate_ctf_and_aux_from_complex(spectrum)
        actual, permutation = model.paired_pit_loss(ctf, direct, reverb)
        expected, expected_permutation = reference.paired_pit_loss(ctf, direct, reverb)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(permutation, expected_permutation)
        swapped_loss, _ = model.paired_pit_loss(ctf.flip(1), direct, reverb)
        torch.testing.assert_close(actual, swapped_loss)
        estimated = model.estimate_ctf(batch["speech_mix"][0])
        assert estimated.shape == (2, 33, 5)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        clone = _sweep_model(room_path_factorized=True).eval()
        clone.load_state_dict(torch.load(buffer), strict=True)
        torch.testing.assert_close(estimated, clone.estimate_ctf(batch["speech_mix"][0]))
        rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
        assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()
    # A late CTF tap must still receive supervision from the full response.
    late_target = torch.zeros_like(reverb)
    late_target[:, 0, 20] = 3
    late_target[:, 1, 158] = -1
    wrong = torch.zeros(2, 2, 33, 5, dtype=torch.complex64, requires_grad=True)
    full_loss, _ = model.paired_pit_loss(wrong, direct, late_target)
    full_loss.backward()
    assert full_loss > 0 and wrong.grad[..., -1].abs().sum() > 0


def test_invalid_code_shapes_and_incompatible_decoder_options():
    model = _predictor()
    case = unittest.TestCase()
    for invalid in (torch.randn(2, 7, 33), torch.randn(2, 1, 7, 33, dtype=torch.complex64),
                    torch.randn(2, 7, 32, dtype=torch.complex64),
                    torch.empty(2, 0, 33, dtype=torch.complex64), None):
        with case.assertRaises(ValueError):
            model.encode_room_path(invalid)
    room, path = torch.randn(2, 33, 8), torch.randn(2, 2, 33, 8)
    invalid_pairs = ((room[:, None], path), (room[..., :7], path),
                     (room, path[:, :, :32]), (room, path[:1]), (room, path[..., :7]),
                     (room, path[:, :0]), (room.to(torch.complex64), path),
                     (room, path.to(torch.complex64)), (room, path.double()), (None, path))
    for invalid_room, invalid_path in invalid_pairs:
        with case.assertRaises(ValueError):
            model.decode_room_path(invalid_room, invalid_path)
    for options in ({"frequency_refinement_position": "before_pool"},
                    {"post_interaction": "speaker"}, {"post_interaction": "joint"}):
        with case.assertRaises(ValueError):
            _predictor(**options)


def test_paired_sweep_loss_retains_complete_convolution_tail():
    """Check every PIT pair against an explicit full-length RIMag cost."""
    model = _sweep_model(room_path_factorized=True).eval()
    direct = torch.zeros(2, 2, 256)
    direct[:, 0, 20] = 1
    direct[:, 1, 30] = -2
    reverb = direct.clone()
    reverb[:, :, 250] = 0.5
    ctf = torch.zeros(2, 2, 33, 5, dtype=torch.complex64)
    ctf[:, 0, :, -1] = 0.7
    ctf[:, 1, :, -1] = -0.4
    excitation = model.response_spectrum(direct.reshape(4, 256))[:, 0]
    target = model.response_spectrum(reverb.reshape(4, 256))[:, 0]
    excitation = excitation.reshape(2, 2, 33, -1)
    target = target.reshape(2, 2, 33, -1)
    expected_frames = excitation.shape[-1] + ctf.shape[-1] - 1
    costs = torch.zeros(2, 2, 2)
    for prediction in (0, 1):
        for reference in (0, 1):
            output = model._complex_convolve(excitation[:, reference], ctf[:, prediction])
            assert output.shape[-1] == expected_frames
            padded_target = F.pad(target[:, reference], (0, expected_frames - target.shape[-1]))
            independent_error = ((output.real - padded_target.real).abs()
                                 + (output.imag - padded_target.imag).abs()
                                 + (output.abs() - padded_target.abs()).abs())
            costs[prediction, reference] = independent_error.flatten(1).mean(1)
    identity = (costs[0, 0] + costs[1, 1]) / 2
    exchanged = (costs[0, 1] + costs[1, 0]) / 2
    expected = torch.stack((identity, exchanged)).amin(0).mean()
    with patch.object(model, "_complex_loss", wraps=model._complex_loss) as observed:
        actual, _ = model.paired_pit_loss(ctf, direct, reverb)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert observed.call_count == 4
    for call in observed.call_args_list:
        assert call.args[0].shape[-1] == expected_frames
        assert call.args[1].shape[-1] == expected_frames


def test_room_path_config_changes_only_the_predictor_flag():
    configs = Path(__file__).resolve().parents[4] / "egs2/whamr/rir_2spk/conf/tuning"
    prefix = "train_pooled_bimamba_2spk_nf_16k_sweep_v2"
    baseline = yaml.safe_load((configs / f"{prefix}.yaml").read_text())
    actual = yaml.safe_load((configs / f"{prefix}_room_path.yaml").read_text())
    expected = copy.deepcopy(baseline)
    expected["model_conf"]["predictor_conf"]["room_path_factorized"] = True
    assert actual == expected


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for key, value in list(globals().items()) if key.startswith("test_"))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
