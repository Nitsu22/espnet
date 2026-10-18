"""CPU checks for full-response waveform supervision and shared two-source PIT."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from espnet2.rir.rec_rir.tflocoformer_sweep_v2_pit import (
    ESPnetTFLocoformerSweepV2PITModel,
)


def model(**kwargs):
    return ESPnetTFLocoformerSweepV2PITModel(
        sr=16000,
        n_fft=64,
        win_len=64,
        hop_len=32,
        num_freqs=33,
        ctf_taps=5,
        n_layers=1,
        emb_dim=8,
        num_groups=2,
        n_heads=2,
        attention_dim=8,
        pos_enc="nope",
        ffn_hidden_dim=[8, 8],
        conv1d_kernel=3,
        pim_sweep_duration=0.1,
        **kwargs,
    )


def paired_example():
    direct = torch.zeros(2, 2, 256)
    direct[:, 0, 20] = 1
    direct[:, 1, 30] = -2
    target = torch.zeros_like(direct)
    target[:, 0, 20] = 3
    target[:, 1, 158] = -1
    ctf = torch.zeros(2, 2, 33, 5, dtype=torch.complex64)
    ctf[:, 0, :, 0] = 3
    ctf[:, 1, :, 4] = 0.5
    return ctf, direct, target


def test_per_example_perfect_gain_and_polarity():
    m = model(time_sweep_loss_weight=1)
    wave = torch.zeros(2, 512)
    wave[:, 64] = torch.tensor([1.0, 5.0])
    wave[:, 192] = torch.tensor([-0.25, 2.0])
    target = m.transforms.stft(wave, "complex")

    # Absolute waveform error retains gain and polarity information, and the
    # target's L1 norm normalizes each example separately.
    torch.testing.assert_close(m.time_sweep_l1(target, target), torch.zeros(2))
    torch.testing.assert_close(m.time_sweep_l1(2 * target, target), torch.ones(2))
    torch.testing.assert_close(m.time_sweep_l1(-target, target), 2 * torch.ones(2))


def test_joint_scaling_invariance():
    m = model(time_sweep_loss_weight=1)
    target_wave = torch.zeros(2, 512)
    target_wave[:, 64] = torch.tensor([1.0, -2.0])
    prediction_wave = target_wave.clone()
    prediction_wave[:, 192] = torch.tensor([0.5, 3.0])
    target = m.transforms.stft(target_wave, "complex")
    prediction = m.transforms.stft(prediction_wave, "complex")
    expected = torch.tensor([0.5, 1.5])
    torch.testing.assert_close(m.time_sweep_l1(prediction, target), expected)
    for scale in (0.125, -4.0):
        torch.testing.assert_close(
            m.time_sweep_l1(scale * prediction, scale * target), expected
        )


def test_full_target_and_prediction_tails_receive_loss_and_gradient():
    m = model(time_sweep_loss_weight=1)
    short_wave = torch.zeros(1, 256)
    short_wave[:, 64] = 1
    long_wave = F.pad(short_wave, (0, 256))
    long_wave[:, 352] = 1
    short = m.transforms.stft(short_wave, "complex")
    long = m.transforms.stft(long_wave, "complex")

    # The same omitted impulse is half of the long target's L1 mass, or all of
    # the short target's mass when it is an unwanted prediction tail.
    torch.testing.assert_close(m.time_sweep_l1(short, long), torch.tensor([0.5]))
    prediction = long.detach().requires_grad_()
    loss = m.time_sweep_l1(prediction, short)
    torch.testing.assert_close(loss, torch.ones(1))
    loss.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[..., short.shape[-1] :].abs().sum() > 0


def test_silent_reference_uses_epsilon_and_has_finite_backward():
    m = model(time_sweep_loss_weight=1, time_sweep_loss_eps=1e-4)
    wave = torch.zeros(2, 256)
    wave[1, 64] = 1e-6
    prediction = m.transforms.stft(wave, "complex").detach().requires_grad_()
    target = torch.zeros_like(prediction)
    loss = m.time_sweep_l1(prediction, target)
    torch.testing.assert_close(loss, torch.tensor([0.0, 0.01]), atol=1e-7, rtol=1e-5)
    loss.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[1].abs().sum() > 0


def test_pit_per_example_assignment_and_reference_pair_swap():
    m = model(time_sweep_loss_weight=0.4)
    ctf, direct, target = paired_example()
    # Waveform supervision compares sweep responses directly; inverse RIR
    # recovery must never participate in this objective.
    with patch.object(m, "response_to_rir", side_effect=AssertionError("inverse used")):
        loss, best, stats = m.paired_pit_loss_with_stats(ctf, direct, target)
        assert loss < 1e-5
        assert stats["loss_time_sweep_l1"] < 1e-5
        torch.testing.assert_close(best, torch.tensor([0, 0]))

        swapped_loss, swapped_best, _ = m.paired_pit_loss_with_stats(
            ctf.flip(1), direct, target
        )
        torch.testing.assert_close(swapped_loss, loss)
        torch.testing.assert_close(swapped_best, torch.tensor([1, 1]))

        swapped_direct, swapped_target = direct.clone(), target.clone()
        swapped_direct[0] = direct[0].flip(0)
        swapped_target[0] = target[0].flip(0)
        paired_loss, paired_best, _ = m.paired_pit_loss_with_stats(
            ctf, swapped_direct, swapped_target
        )
        torch.testing.assert_close(paired_loss, loss)
        torch.testing.assert_close(paired_best, torch.tensor([1, 0]))


def test_combined_pit_selects_one_shared_assignment_and_component_stats():
    m = model(time_sweep_loss_weight=0.5)
    ctf, direct, target = paired_example()
    # Spectral costs prefer identity in both examples. Time costs prefer a
    # swap, strongly enough to change the shared winner only in example zero.
    spectral_costs = [
        torch.tensor(x)
        for x in ([0.0, 0.0], [3.0, 3.0], [3.0, 3.0], [0.0, 0.0])
    ]
    time_costs = [
        torch.tensor(x)
        for x in ([8.0, 4.0], [0.0, 0.0], [0.0, 0.0], [8.0, 4.0])
    ]
    with patch.object(m, "_complex_loss", side_effect=spectral_costs), patch.object(
        m, "time_sweep_l1", side_effect=time_costs
    ):
        loss, best, stats = m.paired_pit_loss_with_stats(ctf, direct, target)
    torch.testing.assert_close(best, torch.tensor([1, 0]))
    torch.testing.assert_close(loss, torch.tensor(2.5))
    torch.testing.assert_close(stats["loss_sweep"], torch.tensor(1.5))
    # Baseline-comparable validation minimizes the spectral component alone;
    # it differs from spectral error under the joint training assignment.
    torch.testing.assert_close(stats["loss_sweep_min"], torch.tensor(0.0))
    assert stats["loss_sweep_min"] < stats["loss_sweep"]
    torch.testing.assert_close(stats["loss_time_sweep_l1"], torch.tensor(2.0))
    torch.testing.assert_close(
        loss, stats["loss_sweep"] + 0.5 * stats["loss_time_sweep_l1"]
    )
    # Independently minimizing the two components would incorrectly return 0.
    assert loss > 0


def test_zero_weight_exactly_preserves_spectral_objective_and_gradient():
    m = model(time_sweep_loss_weight=0)
    ctf, direct, target = paired_example()
    ctf = (ctf + torch.tensor(0.13 + 0.07j)).requires_grad_()
    batch, speakers, samples = direct.shape
    excitation = m.response_spectrum(direct.reshape(batch * speakers, samples))
    reference = m.response_spectrum(target.reshape(batch * speakers, samples))
    excitation = excitation[:, 0].reshape(batch, speakers, *excitation.shape[-2:])
    reference = reference[:, 0].reshape(batch, speakers, *reference.shape[-2:])
    legacy_costs = []
    for permutation in ((0, 1), (1, 0)):
        terms = []
        for pred, ref in enumerate(permutation):
            output = m._complex_convolve(excitation[:, ref], ctf[:, pred])
            frames = max(output.shape[-1], reference.shape[-1])
            terms.append(m._complex_loss(
                F.pad(output, (0, frames - output.shape[-1])),
                F.pad(reference[:, ref], (0, frames - reference.shape[-1])),
                reduction="none",
            ))
        legacy_costs.append((terms[0] + terms[1]) / 2)
    legacy_costs = torch.stack(legacy_costs)
    expected, expected_best = legacy_costs.min(0)
    expected = expected.mean()
    expected_gradient, = torch.autograd.grad(expected, ctf)

    with patch.object(m, "time_sweep_l1", side_effect=AssertionError("zero-weight work")):
        loss, best, stats = m.paired_pit_loss_with_stats(ctf, direct, target)
    gradient, = torch.autograd.grad(loss, ctf)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(best, expected_best, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(stats["loss_sweep"], expected, rtol=0, atol=0)
    assert "loss_time_sweep_l1" not in stats


def test_forward_time_loss_backward_is_finite():
    torch.manual_seed(19)
    m = model(time_sweep_loss_weight=0.3)
    _, direct, target = paired_example()
    speech = torch.randn(2, 256)
    loss, stats, count = m(
        speech_mix=speech,
        speech_mix_lengths=torch.tensor([256, 256]),
        rir_ref1=target[:, 0],
        rir_ref2=target[:, 1],
        rir_direct1=direct[:, 0],
        rir_direct2=direct[:, 1],
    )
    loss.backward()
    gradients = [p.grad for p in m.parameters() if p.grad is not None]
    assert count == 2
    assert torch.isfinite(loss).all()
    assert torch.isfinite(stats["loss_time_sweep_l1"]).all()
    torch.testing.assert_close(
        loss.detach(), stats["loss_sweep"] + 0.3 * stats["loss_time_sweep_l1"]
    )
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_invalid_time_loss_settings_are_rejected():
    check = unittest.TestCase()
    for weight in (-1.0, float("nan"), float("inf"), -float("inf")):
        with check.assertRaises(ValueError):
            model(time_sweep_loss_weight=weight)
    for eps in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
        with check.assertRaises(ValueError):
            model(time_sweep_loss_eps=eps)


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in list(globals().items())
        if name.startswith("test_")
    )
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
