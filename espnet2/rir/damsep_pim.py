"""Ordinary sweep measurement of DAMSEP's causal, direct-to-reverb CTF."""

import numpy as np
import torch
from scipy.signal import fftconvolve

from espnet2.rir.rec_rir.feature import RecRIRTransforms
from espnet2.rir.rec_rir.pim import RecRIRPIM


def causal_filter_spectrum(spectrum, ctf):
    """Full linear frame convolution; coefficient zero is the current frame.

    RecRIRPIM's unfold/matmul instead expects reverse-ordered coefficients.
    FFT convolution here implements DAMSEP's training convention directly,
    including its tail, without changing either model's training code.
    """
    if (
        spectrum.ndim != 2
        or ctf.ndim != 2
        or spectrum.shape[0] != ctf.shape[0]
        or not torch.is_complex(spectrum)
        or not torch.is_complex(ctf)
        or not torch.isfinite(spectrum).all()
        or not torch.isfinite(ctf).all()
    ):
        raise ValueError("Expected finite complex [frequency, frame/tap] tensors")
    length = spectrum.shape[-1] + ctf.shape[-1] - 1
    nfft = 1 << (length - 1).bit_length()
    return torch.fft.ifft(
        torch.fft.fft(spectrum, n=nfft) * torch.fft.fft(ctf, n=nfft), n=nfft
    )[..., :length]


class DAMSEPOrdinaryPIM:
    """Use the reconstruction-loss STFT: FFT512/window256/hop128/sqrt-Hann."""

    def __init__(self):
        self.pim = RecRIRPIM(sr=8000, sweep_duration=8.192)
        self.transforms = RecRIRTransforms(
            sr=8000, n_fft=512, win_len=256, hop_len=128, win_type="sqrthann"
        )
        self.spectrum = self.transforms.stft(self.pim.sinesweep)

    @torch.no_grad()
    def __call__(self, ctf):
        ctf = torch.as_tensor(ctf, device="cpu")
        if ctf.shape != (257, 60) or not torch.is_complex(ctf):
            raise ValueError("Expected DAMSEP complex CTF [257, 60]")
        response = self.transforms.istft(causal_filter_spectrum(self.spectrum, ctf))
        # Linear FFT convolution avoids the quadratic CPU cost of convolve.
        recovered = fftconvolve(
            self.pim.invfilter.numpy().astype(np.float64),
            response.numpy().astype(np.float64),
        )
        start = max(0, int(np.argmax(np.abs(recovered))) - 20)
        recovered = recovered[start:]
        magnitude = float(np.max(np.abs(recovered)))
        if not np.isfinite(recovered).all() or magnitude == 0:
            raise ValueError("Invalid ordinary-PIM response")
        if magnitude > 1:
            recovered /= magnitude
        return np.pad(recovered[:16000], (0, max(0, 16000 - len(recovered)))).astype(
            np.float32
        )
