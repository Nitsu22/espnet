"""Independent waveform checks for oracle-CD -> CA evaluation."""

import unittest

import numpy as np
from scipy.signal import fftconvolve
import torch

from espnet2.rir.rec_rir.ca_evaluation import CAReconstructor
from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.pim import RecRIRPIM


def reconstruction(rir_length=2048, sweep_duration=0.05):
    transform = RecRIRTransforms(8000, 256, 128, "sqrthann", 256)
    pim = RecRIRPIM(8000, sweep_duration)
    return CAReconstructor(pim, transform, rir_length), pim


def physical_measurement(pim, rir, length):
    """Independent full waveform convolution, before any STFT or CTF code."""
    sweep = pim.sinesweep.numpy().astype(np.float64)
    inverse = pim.invfilter.numpy().astype(np.float64)
    gain = fftconvolve(sweep, inverse)[len(sweep) - 1]
    measured = fftconvolve(fftconvolve(sweep, np.asarray(rir)), inverse / gain)
    measured = measured[len(sweep) - 1:]
    return np.pad(measured, (0, max(0, length - len(measured))))[:length]


def pim_taps(causal):
    """The public estimate_ctf() interface reverses causal coefficients."""
    return causal.flip(-1)


def direct_fixture():
    direct = torch.zeros(2, 512)
    direct[0, 73], direct[0, 101] = 0.6, -0.12
    direct[1, 219], direct[1, 247] = -1.8, -0.27
    return direct


def test_identity_matches_independent_original_sweep_measurement():
    # Exercise the actual 8.192-second, 8 kHz measurement as well as the
    # production STFT settings. The reference deliberately stays band-limited.
    recon, pim = reconstruction(sweep_duration=8.192)
    direct = direct_fixture()
    identity = torch.ones(1, 129, 1, dtype=torch.complex64)
    actual = recon.candidate_rirs(identity, direct)[0]
    expected = np.stack([
        physical_measurement(pim, value.numpy(), recon.rir_length)
        for value in direct
    ])
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-4)
    assert actual.abs().argmax(-1).tolist() == [73, 219]
    # A known sweep/inverse colors the RIR: the raw impulse response is a
    # separate evaluation reference, not an exact identity reconstruction.
    assert np.linalg.norm(expected[:, :512] - direct.numpy()) > 1e-4


def test_tap_reversal_and_hop_delay_match_waveform_convolution():
    recon, pim = reconstruction()
    direct = direct_fixture()
    causal = torch.zeros(1, 129, 5, dtype=torch.complex64)
    causal[..., 4] = 1
    actual = recon.candidate_rirs(pim_taps(causal), direct)[0]
    delay = 4 * recon.transform.hop_len
    expected = np.stack([
        physical_measurement(pim, np.pad(value.numpy(), (delay, 0)), recon.rir_length)
        for value in direct
    ])
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-4)
    assert actual.abs().argmax(-1).tolist() == [73 + delay, 219 + delay]


def test_prediction_gain_polarity_and_cd_amplitude_are_preserved():
    recon, pim = reconstruction()
    direct = direct_fixture()
    causal = torch.zeros(2, 129, 3, dtype=torch.complex64)
    causal[0, :, 0], causal[1, :, 0] = 0.3, -3.0
    actual = recon.candidate_rirs(pim_taps(causal), direct)
    baseline = np.stack([
        physical_measurement(pim, value.numpy(), recon.rir_length)
        for value in direct
    ])
    expected = np.stack([0.3 * baseline, -3.0 * baseline])
    np.testing.assert_allclose(actual.numpy(), expected, atol=3e-6, rtol=3e-4)
    assert actual[1, 1, 219] > 5  # No predicted-peak normalization or gain fit.
    assert actual[1, 0, 73] < 0
    assert actual[0, 1, 219] < 0


def test_full_excitation_and_delayed_prediction_tails_are_kept():
    recon, pim = reconstruction(rir_length=2560)
    direct = torch.zeros(1, 512)
    direct[0, -1] = 1
    # This reflection arrives after the complete unfiltered CD-sweep response.
    # Cropping the excitation or causal convolution to input length loses it.
    causal = torch.zeros(1, 129, 9, dtype=torch.complex64)
    causal[..., 0], causal[..., 8] = 1, 0.7
    actual = recon.candidate_rirs(pim_taps(causal), direct)[0, 0]
    reference = np.zeros(512 + 8 * recon.transform.hop_len)
    reference[511], reference[-1] = 1, 0.7
    expected = physical_measurement(pim, reference, recon.rir_length)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-4)
    assert abs(actual[511 + 8 * recon.transform.hop_len].item() - 0.7) < 1e-5
    assert actual[1400:].abs().sum() > 0.5


def test_cd_reordering_only_reorders_candidate_reference_axis():
    recon, _ = reconstruction()
    direct = direct_fixture()
    causal = torch.zeros(2, 129, 4, dtype=torch.complex64)
    causal[0, :, 0], causal[0, :, 3] = 1, 0.2
    causal[1, :, 1], causal[1, :, 2] = -0.8, 0.1
    original = recon.candidate_rirs(pim_taps(causal), direct)
    reordered = recon.candidate_rirs(pim_taps(causal), direct.flip(0))
    assert original.shape == (2, 2, recon.rir_length)
    torch.testing.assert_close(reordered, original.flip(1), atol=1e-6, rtol=1e-5)
    # All slot/reference combinations are retained for one later PIT choice.
    assert not torch.allclose(original[0, 0], original[0, 1])
    assert not torch.allclose(original[0, 0], original[1, 0])


def test_zero_cd_and_incompatible_frequency_count_are_rejected():
    recon, _ = reconstruction()
    with unittest.TestCase().assertRaisesRegex(ValueError, "nonzero"):
        recon.candidate_rirs(torch.ones(1, 129, 1, dtype=torch.complex64),
                             torch.zeros(2, 512))
    with unittest.TestCase().assertRaisesRegex(ValueError, "frequency count"):
        recon.candidate_rirs(torch.ones(1, 33, 1, dtype=torch.complex64),
                             direct_fixture())


if __name__ == "__main__":
    suite = unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in list(globals().items()) if name.startswith("test_")
    )
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
