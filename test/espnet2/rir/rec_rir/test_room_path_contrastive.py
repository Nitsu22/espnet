"""Pair identity, PIT, auxiliary gradients, baseline compatibility and input checks."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from espnet2.rir.rec_rir.pooled_bimamba_room_path_contrastive import (
    ESPnetPooledBiMambaRoomPathContrastiveModel,
    symmetric_infonce,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor_rec_rir_room_path_contrastive import (
    RecRIRRoomPathContrastivePreprocessor,
)


def _settings():
    return dict(
        sr=16000, n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, pim_sweep_duration=0.1,
        predictor_conf=dict(
            room_path_factorized=True, emb_dim=8, pre_layers=1, post_layers=1,
            d_state=4, d_conv=2, freq_bottleneck=4, freq_rank=4, local_kernel=3,
            pooling_hidden=4, n_heads=2, ffn_dim=8, head_dim=8, time_module="bilstm"),
    )


def _model(**options):
    return ESPnetPooledBiMambaRoomPathContrastiveModel(**(_settings() | options))


def _batch():
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_view2=torch.randn(2, 256),
                 speech_mix_lengths=torch.tensor([256, 192]),
                 speech_mix_view2_lengths=torch.tensor([256, 192]))
    batch["speech_mix"][1, 192:] = 0
    batch["speech_mix_view2"][1, 192:] = 0
    for speaker in (1, 2):
        direct = torch.zeros(2, 256)
        direct[:, 20 + speaker] = speaker
        reverb = direct.clone()
        reverb[:, 125 + 10 * speaker] = 0.3 * speaker
        batch[f"rir_direct{speaker}"] = direct
        batch[f"rir_ref{speaker}"] = reverb
    return batch


def test_infonce_positive_and_negative_identity_matches_analytic_cost():
    # Identical orthogonal paths: one positive and TWO other-path negatives
    # for every anchor. Same-path counterparts must never be negatives.
    temperature = 0.5
    identity = torch.eye(2)[None]
    actual = symmetric_infonce(identity, identity, temperature)
    expected = math.log1p(2 * math.exp(-1 / temperature))
    torch.testing.assert_close(actual, torch.tensor(expected))
    wrong = symmetric_infonce(identity, identity.flip(1), temperature)
    assert wrong > actual + 1
    # Room loss with one room has no negative and must be finite exact zero.
    singleton = torch.randn(1, 1, 5, requires_grad=True)
    zero = symmetric_infonce(singleton, singleton * 2, temperature)
    assert zero == 0
    zero.backward()
    assert singleton.grad is not None and torch.isfinite(singleton.grad).all()


def test_path_negatives_stay_in_their_room_and_views_are_symmetric():
    torch.manual_seed(701)
    first = torch.randn(3, 2, 7, requires_grad=True)
    second = torch.randn(3, 2, 7, requires_grad=True)
    combined = symmetric_infonce(first, second, 0.2)
    separate = torch.stack([
        symmetric_infonce(first[i:i+1], second[i:i+1], 0.2) for i in range(3)
    ]).mean()
    torch.testing.assert_close(combined, separate)
    torch.testing.assert_close(combined, symmetric_infonce(second, first, 0.2))
    combined.backward()
    assert all(value.grad.abs().sum() > 0 and torch.isfinite(value.grad).all()
               for value in (first, second))


def test_predictor_pre_slot_codes_are_optional_and_do_not_change_decoder():
    torch.manual_seed(703)
    predictor = _model().ctf_predictor.eval()
    spectrum = torch.randn(2, 9, 33, dtype=torch.complex64)
    before, _, auxiliary = predictor(spectrum)
    after, _, extra = predictor(spectrum, additional={"return_pre_slot_codes": True})
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert list(auxiliary) == ["room_code", "path_codes"]
    torch.testing.assert_close(
        extra["pre_slot_path_codes"] + predictor.slot_embeddings[None, :, None],
        extra["path_codes"], rtol=0, atol=0,
    )


def test_full_paired_forward_backward_and_pit_slot_permutation_invariance():
    torch.manual_seed(709)
    model = _model(projection_dim=4).train()
    batch = _batch()
    loss, stats, weight = model(**batch)
    assert weight == 2 and torch.isfinite(loss)
    torch.testing.assert_close(loss, stats["loss_sweep"] + 0.01 * stats["loss_room"]
                               + 0.01 * stats["loss_path"])
    assert stats["room_negative_count"] == 2
    loss.backward()
    for component in ("room_projection", "path_projection", "ctf_predictor.encoder",
                      "ctf_predictor.room_pooling", "ctf_predictor.pooling"):
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith(component) and p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0, component
    # Independently exchange prediction slots in the second view. Its CTF PIT
    # assignment must exchange as well, leaving both contrastive losses intact.
    estimate = model._estimate_ctf_and_aux_from_complex

    def swapped(spectrum):
        ctf, auxiliary = estimate(spectrum)
        ctf = torch.cat((ctf[:2], ctf[2:].flip(1)), 0)
        for name in ("path_codes", "pre_slot_path_codes"):
            value = auxiliary[name]
            auxiliary[name] = torch.cat((value[:2], value[2:].flip(1)), 0)
        return ctf, auxiliary

    with patch.object(model, "_estimate_ctf_and_aux_from_complex", swapped):
        switched, switched_stats, _ = model(**batch)
    torch.testing.assert_close(loss, switched, rtol=1e-6, atol=1e-6)
    for name in ("loss_sweep", "loss_path", "loss_room"):
        torch.testing.assert_close(stats[name], switched_stats[name], rtol=1e-6, atol=1e-6)
    # Exchanging the teacher path labels in BOTH views cannot change training.
    relabeled = dict(batch)
    for prefix in ("rir_direct", "rir_ref"):
        relabeled[prefix + "1"] = batch[prefix + "2"]
        relabeled[prefix + "2"] = batch[prefix + "1"]
    relabeled_loss, _, _ = model(**relabeled)
    torch.testing.assert_close(loss, relabeled_loss, rtol=1e-6, atol=1e-6)


def test_zero_weight_control_and_single_view_validation_match_baseline():
    torch.manual_seed(719)
    baseline = ESPnetPooledBiMambaSweepV2PITModel(**_settings()).eval()
    torch.manual_seed(719)
    model = _model(lambda_room=0, lambda_path=0).train()
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)
    batch = _batch()
    paired_loss, stats, _ = model(**batch)
    first = dict(batch)
    second = dict(batch, speech_mix=batch["speech_mix_view2"],
                  speech_mix_lengths=batch["speech_mix_view2_lengths"])
    first_loss, _, _ = baseline(**first)
    second_loss, _, _ = baseline(**second)
    torch.testing.assert_close(paired_loss, (first_loss + second_loss) / 2,
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(paired_loss, stats["loss_sweep"], rtol=0, atol=0)
    model.eval()
    del first["speech_mix_view2"], first["speech_mix_view2_lengths"]
    validation_loss, validation_stats, _ = model(**first)
    torch.testing.assert_close(validation_loss, first_loss, rtol=0, atol=0)
    torch.testing.assert_close(validation_stats["loss_sweep"], first_loss, rtol=0, atol=0)
    model.validation_contrastive = True
    diagnostic_loss, diagnostic_stats, _ = model(**batch)
    torch.testing.assert_close(diagnostic_loss, first_loss, rtol=1e-6, atol=1e-6)
    assert "loss_room" in diagnostic_stats and "loss_path" in diagnostic_stats
    # PIM and regular evaluation need only the primary waveform, not view2.
    with torch.no_grad():
        waveform = batch["speech_mix"][0]
        torch.testing.assert_close(model.estimate_ctf(waveform), baseline.estimate_ctf(waveform),
                                   rtol=0, atol=0)
        actual_rirs = model.estimate_rir(waveform, rir_length=256)
        torch.testing.assert_close(actual_rirs, baseline.estimate_rir(waveform, rir_length=256),
                                   rtol=0, atol=0)
        assert actual_rirs.shape == (2, 256) and torch.isfinite(actual_rirs).all()


def test_preprocessor_independent_crops_shared_rirs_and_common_collation():
    processor = RecRIRRoomPathContrastivePreprocessor(True, speech_segment=128, rir_length=256)
    data = {"speech_mix": np.arange(1, 257, dtype=float),
            "speech_mix_view2": np.arange(1001, 1257, dtype=float)}
    for speaker in (1, 2):
        direct = np.zeros((256, 2))
        direct[20 + speaker, 0] = speaker
        reverb = direct.copy()
        reverb[150 + speaker, 0] = speaker / 4
        data[f"rir_direct{speaker}"] = direct
        data[f"rir_ref{speaker}"] = reverb
    with patch("numpy.random.randint", side_effect=[3, 11]) as choose:
        prepared = processor("pair", data)
    assert choose.call_count == 2
    np.testing.assert_array_equal(prepared["speech_mix"], data["speech_mix"][3:131])
    np.testing.assert_array_equal(prepared["speech_mix_view2"], data["speech_mix_view2"][11:139])
    for speaker in (1, 2):
        assert prepared[f"rir_direct{speaker}"][20 + speaker] == 1
        assert prepared[f"rir_ref{speaker}"][150 + speaker] == 0.25
    short = dict(prepared, speech_mix=prepared["speech_mix"][:80],
                 speech_mix_view2=prepared["speech_mix_view2"][:80])
    ids, batch = CommonCollateFn(float_pad_value=0.0, int_pad_value=0)(
        [("long", prepared), ("short", short)])
    assert ids == ["long", "short"]
    assert batch["speech_mix_lengths"].tolist() == [128, 80]
    assert batch["speech_mix_view2_lengths"].tolist() == [128, 80]
    assert batch["speech_mix_view2"][1, 80:].count_nonzero() == 0
    processor.train = False
    full = processor("pair", data)
    assert len(full["speech_mix"]) == len(full["speech_mix_view2"]) == 256


def test_bad_settings_missing_view_and_inconsistent_rir_lengths_fail():
    case = unittest.TestCase()
    for options in ({"lambda_room": -1}, {"lambda_path": float("nan")},
                    {"contrastive_temperature": 0}, {"projection_dim": 0},
                    {"validation_contrastive": "false"}):
        with case.assertRaises(ValueError):
            _model(**options)
    model = _model().train()
    batch = _batch()
    broken = dict(batch)
    del broken["speech_mix_view2"]
    with case.assertRaisesRegex(ValueError, "speech_mix_view2"):
        model(**broken)
    with case.assertRaisesRegex(ValueError, "lengths"):
        model(**dict(batch, speech_mix_view2_lengths=torch.tensor([256, 257])))
    with case.assertRaisesRegex(ValueError, "fixed-length"):
        model(**dict(batch, rir_direct1_lengths=torch.tensor([256, 100])))


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
                               for key, value in list(globals().items()) if key.startswith("test_"))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
