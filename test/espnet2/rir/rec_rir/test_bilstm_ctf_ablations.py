"""CPU checks of independent and combined CTF architecture/loss variants."""

import copy
import io
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from espnet2.rir.rec_rir.pooled_bimamba import (
    FramewiseFrequencyBlock,
    FrequencyBiLSTMBlock,
    LightFrequencyBlock,
    PooledBiMambaCTFPredictor,
    SpeakerQueryRefinement,
)
from espnet2.rir.rec_rir.pooled_bimamba_sweep_v2_pit import (
    ESPnetPooledBiMambaSweepV2PITModel,
)
from test.espnet2.rir.rec_rir.test_sweep_v2_drr_pit import fixture


REPO = Path(__file__).resolve().parents[4]
RECIPE = REPO / "egs2/whamr/rir_2spk"
PREFIX = "train_pooled_bilstm_2spk_nf_8k_sweep_v2_pre8_post4_"
BASELINE = RECIPE / "conf/tuning" / (PREFIX + "time_l1_plateau_200ep.yaml")
ARCHITECTURES = {
    "pre_last2_bilstm": {
        "pre_frequency_modules": ["light"] * 6 + ["bilstm"] * 2,
    },
    "rank32": {"freq_rank": 32},
    "speaker_query": {"pooling_mode": "speaker_query"},
}
VARIANTS = tuple(ARCHITECTURES) + ("rir_mse",) + tuple(
    name + "_rir_mse" for name in ARCHITECTURES
)


def config(variant):
    path = BASELINE if variant == "baseline" else (
        RECIPE / "conf/tuning" / (PREFIX + variant + ".yaml")
    )
    return yaml.safe_load(path.read_text())


def small_model(variant="baseline", **changes):
    conf = copy.deepcopy(config(variant)["model_conf"])
    conf.update(
        n_fft=64, win_len=64, hop_len=32, num_freqs=33,
        ctf_taps=5, pim_sweep_duration=0.1,
    )
    conf.update(changes)
    conf["predictor_conf"].update(
        emb_dim=8, freq_bottleneck=4, local_kernel=3, pooling_hidden=4,
        n_heads=2, ffn_dim=8, head_dim=8,
    )
    with patch("espnet2.rir.rec_rir.pooled_bimamba._build_mamba",
               side_effect=AssertionError("BiLSTM variants must not build Mamba")):
        return ESPnetPooledBiMambaSweepV2PITModel(**conf)


def test_configs_and_launcher_change_only_requested_factors():
    baseline = config("baseline")
    launcher = RECIPE / "run_pooled_bilstm_ctf_ablations_8k.sh"
    for variant in ("baseline",) + VARIANTS:
        expected = copy.deepcopy(baseline)
        arch = variant.removesuffix("_rir_mse")
        expected["model_conf"]["predictor_conf"].update(ARCHITECTURES.get(arch, {}))
        if variant.endswith("rir_mse"):
            expected["model_conf"].update(
                time_sweep_loss_weight=0.0,
                rir_mse_loss_weight=10000.0,
                rir_mse_loss_eps=1e-8,
            )
        assert config(variant) == expected
        path = subprocess.check_output(
            ["bash", str(launcher), variant, "--print-config"], text=True,
        ).strip()
        assert yaml.safe_load(Path(path).read_text()) == expected
    result = subprocess.run(
        ["bash", str(launcher), "typo", "--print-config"], capture_output=True,
    )
    assert result.returncode == 2


def test_last_two_pre_mixers_and_shared_initialization():
    torch.manual_seed(42)
    baseline = small_model().ctf_predictor
    torch.manual_seed(42)
    modified = small_model("pre_last2_bilstm").ctf_predictor
    assert all(isinstance(b, LightFrequencyBlock) for b in modified.frequency_blocks[:6])
    assert all(isinstance(b, FramewiseFrequencyBlock) for b in modified.frequency_blocks[6:])
    assert all(isinstance(b.sequence_block, FrequencyBiLSTMBlock)
               for b in modified.frequency_blocks[6:])
    for name, value in baseline.state_dict().items():
        if not name.startswith(("frequency_blocks.6.", "frequency_blocks.7.")):
            torch.testing.assert_close(value, modified.state_dict()[name], rtol=0, atol=0)
    seen = []
    handles = [block.sequence_block.lstm.register_forward_pre_hook(
        lambda module, args: seen.append(tuple(args[0].shape)))
        for block in modified.frequency_blocks[6:]]
    result = modified(torch.randn(2, 9, 33, dtype=torch.complex64))[0]
    for handle in handles:
        handle.remove()
    assert seen == [(18, 33, 8), (18, 33, 8)]
    result.abs().square().mean().backward()
    for block in modified.frequency_blocks[6:]:
        for name in ("weight_ih_l0", "weight_ih_l0_reverse"):
            assert getattr(block.sequence_block.lstm, name).grad.abs().sum() > 0


def test_query_pooling_is_conditioned_shared_and_can_recover_legacy():
    torch.manual_seed(42)
    baseline = small_model().ctf_predictor.eval()
    torch.manual_seed(42)
    modified = small_model("speaker_query").ctf_predictor.eval()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, modified.state_dict()[name], rtol=0, atol=0)
    observation = torch.randn(2, 9, 33, dtype=torch.complex64)
    actual = modified(observation)[0]
    assert not torch.allclose(actual, baseline(observation)[0])
    actual.abs().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in modified.query_refinement.parameters())
    assert modified.query_refinement.query.weight.grad.abs().sum() > 0
    assert modified.query_refinement.key.weight.grad.abs().sum() > 0
    # Disabling query logits recovers the original channelwise weighted pooling.
    with torch.no_grad():
        modified.query_refinement.query.weight.zero_()
        torch.testing.assert_close(modified(observation)[0], baseline(observation)[0],
                                   rtol=0, atol=0)

    module = SpeakerQueryRefinement(8, 1e-5)
    x = torch.randn(2, 7, 11, 8)
    pooled = torch.randn(2, 2, 11, 8)
    slots = torch.randn(2, 8)
    scores = module(x, pooled, slots)
    assert scores.shape == (2, 7, 11, 2)
    torch.testing.assert_close(
        module(x, pooled.flip(1), slots.flip(0)), scores.flip(-1),
    )
    changed = pooled.clone()
    changed[:, 0, 4, 0] += 5
    # Context at another frequency must influence this frequency's time scores.
    assert not torch.allclose(module(x, changed, slots)[:, :, 0, 0], scores[:, :, 0, 0])


def test_rir_mse_preserves_gain_delay_polarity_and_teacher_scaling():
    model = small_model("rir_mse")
    target = torch.zeros(2, 256)
    target[:, 30] = torch.tensor([1.0, -3.0])
    target[:, 130] = torch.tensor([0.5, 1.5])
    torch.testing.assert_close(model.rir_peak_mse(target, target), torch.zeros(2))
    expected = torch.full((2,), 1.25 / 256)
    torch.testing.assert_close(model.rir_peak_mse(2 * target, target), expected)
    torch.testing.assert_close(model.rir_peak_mse(-target, target), 4 * expected)
    prediction = target.clone()
    prediction[:, 130] *= 0.3
    loss = model.rir_peak_mse(prediction, target)
    for scale in (0.125, -4.0):
        torch.testing.assert_close(model.rir_peak_mse(scale * prediction, scale * target), loss)
    assert (model.rir_peak_mse(target.roll(1, -1), target) > 0).all()
    tiny = torch.full((2, 256), 1e-10, requires_grad=True)
    silent_loss = model.rir_peak_mse(tiny, torch.zeros_like(tiny)).sum()
    silent_loss.backward()
    assert torch.isfinite(silent_loss) and torch.isfinite(tiny.grad).all()


def test_mse_uses_matched_inverse_measurement_and_paired_pit():
    model = small_model("rir_mse")
    ctf, direct, target = fixture()
    loss, best, stats = model.paired_pit_loss_with_stats(ctf, direct, target)
    assert loss < 1e-5 and stats["loss_rir_mse"] < 1e-8
    torch.testing.assert_close(best, torch.tensor([0, 0]))
    # The band-limited inverse-sweep teacher differs from the raw physical RIR.
    recovered = model.response_to_rir(model.response_spectrum(target[:, 0]), 256)[:, 0]
    assert not torch.allclose(recovered, target[:, 0])
    d2, t2 = direct.clone(), target.clone()
    d2[0], t2[0] = direct[0].flip(0), target[0].flip(0)
    swapped, winner, _ = model.paired_pit_loss_with_stats(ctf, d2, t2)
    torch.testing.assert_close(swapped, loss)
    torch.testing.assert_close(winner, torch.tensor([1, 0]))
    wrong = ctf.clone()
    wrong[:, 0, :, 4] = 0.1
    wrong.requires_grad_()
    response = model._complex_convolve(model.response_spectrum(direct[:, 0])[:, 0], wrong[:, 0])
    rir = model.response_to_rir(response[:, None], 256)[:, 0]
    term = model.rir_peak_mse(rir, recovered).mean()
    term.backward()
    assert term > 0 and torch.isfinite(wrong.grad).all()
    assert wrong.grad[:, 0, :, 4].abs().sum() > 0


def test_mse_and_spectrum_share_one_assignment():
    model = small_model("rir_mse", rir_mse_loss_weight=0.5)
    ctf, direct, target = fixture()
    spectral = [torch.tensor(x) for x in ([0., 0.], [3., 3.], [3., 3.], [0., 0.])]
    mse = [torch.tensor(x) for x in ([8., 4.], [0., 0.], [0., 0.], [8., 4.])]
    with patch.object(model, "_complex_loss", side_effect=spectral), patch.object(
        model, "rir_peak_mse", side_effect=mse,
    ):
        loss, best, stats = model.paired_pit_loss_with_stats(ctf, direct, target)
    torch.testing.assert_close(best, torch.tensor([1, 0]))
    torch.testing.assert_close(loss, torch.tensor(2.5))
    torch.testing.assert_close(loss, stats["loss_sweep"] + 0.5 * stats["loss_rir_mse"])
    assert stats["loss_sweep_min"] == 0 and stats["loss_rir_mse_min"] == 0


def test_disabled_mse_preserves_legacy_state_loss_and_gradient():
    torch.manual_seed(42)
    old = small_model()
    torch.manual_seed(42)
    new = small_model(rir_mse_loss_weight=0)
    assert old.state_dict().keys() == new.state_dict().keys()
    ctf, direct, target = fixture()
    ctf = (ctf + 0.13 + 0.07j).requires_grad_()
    expected = old.paired_pit_loss(ctf, direct, target)[0]
    expected_grad, = torch.autograd.grad(expected, ctf)
    with patch.object(new, "response_to_rir", side_effect=AssertionError("unused inverse")):
        actual = new.paired_pit_loss(ctf, direct, target)[0]
    actual_grad, = torch.autograd.grad(actual, ctf)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    # Loss-only options add no persistent parameters/buffers to checkpoints.
    small_model("rir_mse").load_state_dict(old.state_dict(), strict=True)


def test_every_variant_trains_reloads_and_infers_two_rirs():
    _, direct, target = fixture()
    batch = dict(speech_mix=torch.randn(2, 256), speech_mix_lengths=torch.tensor([256, 256]))
    for speaker in range(2):
        batch[f"rir_direct{speaker + 1}"] = direct[:, speaker]
        batch[f"rir_ref{speaker + 1}"] = target[:, speaker]
    for variant in VARIANTS:
        model = small_model(variant)
        loss, stats, count = model(**batch)
        assert count == 2 and torch.isfinite(loss)
        auxiliary = "loss_rir_mse" if variant.endswith("rir_mse") else "loss_time_sweep_l1"
        weight = (model.rir_mse_loss_weight if variant.endswith("rir_mse")
                  else model.time_sweep_loss_weight)
        torch.testing.assert_close(loss.detach(), stats["loss_sweep"] + weight * stats[auxiliary])
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad), variant
        model.eval()
        with torch.no_grad():
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            clone = small_model(variant).eval()
            clone.load_state_dict(torch.load(buffer), strict=True)
            torch.testing.assert_close(model.estimate_ctf(batch["speech_mix"][0]),
                                       clone.estimate_ctf(batch["speech_mix"][0]))
            rirs = clone.estimate_rir(batch["speech_mix"][0], rir_length=256)
            assert rirs.shape == (2, 256) and torch.isfinite(rirs).all()


def test_optional_losses_can_coexist():
    model = small_model("speaker_query_rir_mse", time_sweep_loss_weight=0.3,
                        rir_mse_loss_weight=0.7, drr_loss_weight=0.1)
    ctf, direct, target = fixture()
    ctf = (ctf + 0.1j).requires_grad_()
    loss, _, stats = model.paired_pit_loss_with_stats(ctf, direct, target)
    expected = (stats["loss_sweep"] + 0.3 * stats["loss_time_sweep_l1"]
                + 0.7 * stats["loss_rir_mse"] + 0.1 * stats["loss_drr"])
    torch.testing.assert_close(loss.detach(), expected)
    loss.backward()
    assert torch.isfinite(ctf.grad).all() and ctf.grad.abs().sum() > 0


def test_invalid_new_options_are_rejected():
    check = unittest.TestCase()
    for value in (-1., float("nan"), float("inf")):
        with check.assertRaises(ValueError):
            small_model(rir_mse_loss_weight=value)
    for value in (0., -1., float("nan"), float("inf")):
        with check.assertRaises(ValueError):
            small_model(rir_mse_loss_eps=value)
    for changes in (
        dict(pooling_mode="typo"),
        dict(pre_frequency_modules="bilstm"),
        dict(pre_frequency_modules=["light"]),
        dict(pre_frequency_modules=["typo"] * 4),
        dict(pre_frequency_module="bilstm", pre_frequency_modules=["light"] * 4),
    ):
        with check.assertRaises(ValueError):
            PooledBiMambaCTFPredictor(**changes)


if __name__ == "__main__":
    torch.set_num_threads(1)
    suite = unittest.TestSuite(unittest.FunctionTestCase(value)
        for name, value in list(globals().items()) if name.startswith("test_"))
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
