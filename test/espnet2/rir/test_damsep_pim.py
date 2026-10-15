import numpy as np
import pytest
import torch
from torch.nn.functional import pad

from espnet2.rir.damsep_pim import DAMSEPOrdinaryPIM, causal_filter_spectrum


def test_causal_full_convolution_and_reversed_rec_rir_order():
    torch.manual_seed(10)
    spectrum = torch.randn(3, 17, dtype=torch.complex64)
    coefficients = torch.randn(3, 5, dtype=torch.complex64)
    actual = causal_filter_spectrum(spectrum, coefficients)
    independent = np.stack(
        [np.convolve(x.numpy(), h.numpy()) for x, h in zip(spectrum, coefficients)]
    )
    np.testing.assert_allclose(actual.numpy(), independent, atol=2e-6, rtol=2e-6)
    rec_rir = (
        pad(spectrum, (4, 4)).unfold(1, 5, 1) @ coefficients.flip(-1).unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(actual, rec_rir, atol=2e-6, rtol=2e-6)


def test_positive_tap_is_positive_frame_delay():
    spectrum = torch.zeros(2, 12, dtype=torch.complex64)
    spectrum[:, 3] = 1
    coefficients = torch.zeros(2, 6, dtype=torch.complex64)
    coefficients[:, 4] = 1
    result = causal_filter_spectrum(spectrum, coefficients)
    assert result.abs().argmax(-1).tolist() == [7, 7]


def test_identity_matches_ordinary_sweep_and_full_delayed_identity():
    decoder = DAMSEPOrdinaryPIM()
    coefficients = np.zeros((257, 60), dtype=np.complex64)
    coefficients[:, 0] = 1
    expected = __import__("scipy").signal.fftconvolve(
        decoder.pim.sinesweep.numpy().astype(np.float64),
        decoder.pim.invfilter.numpy().astype(np.float64),
    )
    start = np.argmax(abs(expected)) - 20
    expected = expected[start : start + 16000]
    if np.max(abs(expected)) > 1:
        expected /= np.max(abs(expected))
    np.testing.assert_allclose(decoder(coefficients), expected, atol=1e-6)
    delayed = np.zeros_like(coefficients)
    delayed[:, 5] = 1
    np.testing.assert_allclose(decoder(delayed), decoder(coefficients), atol=1e-6)
    coefficients[0, 0] = np.nan
    with pytest.raises(ValueError):
        decoder(coefficients)
