"""Finite direct-to-reverberant CTFs fitted to the Sweep v2 measurement.

Coefficients are in causal order: tap zero multiplies the current frame.
The least-squares operator includes the complete predicted convolution tail.
This is an oracle reconstruction reference, not a bound on separation quality.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import fft, linalg


@dataclass(frozen=True)
class OracleCTFConfig:
    sample_rate: int = 8000
    n_fft: int = 256
    hop_length: int = 128
    win_length: int = 256
    window: str = "sqrthann"
    taps: int = 60
    rir_length: int = 16000
    sweep_duration: float = 8.192
    regularization: float = 1.0e-6

    def __post_init__(self):
        if (min(self.sample_rate, self.n_fft, self.hop_length,
                self.win_length, self.taps, self.rir_length) <= 0
                or self.win_length > self.n_fft
                or not np.isfinite(self.sweep_duration)
                or self.sweep_duration <= 0
                or not np.isfinite(self.regularization)
                or self.regularization < 0):
            raise ValueError("Invalid Oracle CTF configuration")

    def as_dict(self):
        return asdict(self)


def normalize_rir_pair(direct, reverb, length):
    """Preserve pair timing/gain, matching RecRIRSweepV2Preprocessor."""
    pair = []
    for value in (direct, reverb):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim == 2:
            value = value[:, 0]
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError("Expected finite, nonempty mono RIRs")
        if np.any(value[length:] != 0):
            raise ValueError("Physical RIR exceeds rir_length; do not truncate its tail")
        pair.append(value)
    scale = np.abs(pair[0]).max()
    if scale == 0 or not np.any(pair[1]):
        raise ValueError("Expected nonzero direct and reverberant RIRs")
    result = []
    for value in pair:
        target = np.zeros(length, dtype=np.float64)
        target[: min(length, len(value))] = value[:length] / scale
        result.append(target)
    return result


def solve_full_convolution_ctf(excitation, target, taps, regularization=1.0e-6):
    """Solve full-convolution ridge LS independently for every frequency.

    For X[n,l] = excitation[n-l], X^H X is Hermitian Toeplitz. FFT
    correlations construct its first column and X^H target without storing
    the large time-by-tap design matrices. The ridge diagonal is relative to
    each frequency's excitation energy; zero-energy bins return zero filters.
    """
    x = np.asarray(excitation, dtype=np.complex128)
    y = np.asarray(target, dtype=np.complex128)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]
            or not x.shape[1] or not y.shape[1] or taps < 1
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(regularization) or regularization < 0):
        raise ValueError("Expected finite [frequency, frame] spectra and valid LS settings")
    n_fft = fft.next_fast_len(max(2 * x.shape[-1] - 1,
                                  x.shape[-1] + y.shape[-1] - 1, taps))
    xf = fft.fft(x, n=n_fft, axis=-1)
    yf = fft.fft(y, n=n_fft, axis=-1)
    auto = fft.ifft(xf * xf.conj(), axis=-1)[:, :taps].copy()
    rhs = fft.ifft(yf * xf.conj(), axis=-1)[:, :taps].copy()
    # Lags beyond the finite excitation/target have no overlap. This also
    # prevents circular wraparound when a caller requests many short-signal taps.
    auto[:, x.shape[-1]:] = 0
    rhs[:, y.shape[-1]:] = 0
    energy = np.sum(np.abs(x) ** 2, axis=-1)
    auto[:, 0] = energy * (1 + regularization)
    coefficients = np.zeros((len(x), taps), dtype=np.complex128)
    for frequency in range(len(x)):
        if energy[frequency] == 0:
            continue
        column = auto[frequency]
        try:
            solution = linalg.solve_toeplitz(
                (column, column.conj()), rhs[frequency], check_finite=False)
        except linalg.LinAlgError:
            # A dense Hermitian solve is a small, robust fallback if Levinson
            # encounters a zero intermediate pivot.
            matrix = linalg.toeplitz(column, column.conj())
            solution = linalg.lstsq(matrix, rhs[frequency])[0]
        if not np.isfinite(solution).all():
            raise ValueError(f"Nonfinite Oracle CTF at frequency {frequency}")
        coefficients[frequency] = solution
    prediction = fft.ifft(
        fft.fft(x, n=fft.next_fast_len(x.shape[-1] + taps - 1), axis=-1)
        * fft.fft(coefficients,
                  n=fft.next_fast_len(x.shape[-1] + taps - 1), axis=-1), axis=-1)
    prediction = prediction[:, : x.shape[-1] + taps - 1]
    frames = max(prediction.shape[-1], y.shape[-1])
    prediction = np.pad(prediction, ((0, 0), (0, frames - prediction.shape[-1])))
    padded_target = np.pad(y, ((0, 0), (0, frames - y.shape[-1])))
    residual = prediction - padded_target
    squared_error = float(np.sum(np.abs(residual) ** 2))
    target_energy = float(np.sum(np.abs(y) ** 2))
    metrics = {
        "squared_error": squared_error,
        "target_energy": target_energy,
        "relative_mse": squared_error / max(target_energy, np.finfo(float).tiny),
        "rimag": float(np.mean(np.abs(residual.real) + np.abs(residual.imag)
                                + np.abs(np.abs(prediction) - np.abs(padded_target)))),
        "zero_excitation_bins": int(np.count_nonzero(energy == 0)),
        "frames": frames,
    }
    return coefficients, metrics


class OracleCTFFitter:
    """Reuse one exact Rec-RIR sweep/STFT for all physical RIR pairs."""

    def __init__(self, config=OracleCTFConfig()):
        import torch

        from espnet2.rir.rec_rir.feature import RecRIRTransforms
        from espnet2.rir.rec_rir.pim import RecRIRPIM

        self.config = config
        self.transforms = RecRIRTransforms(
            sr=config.sample_rate, n_fft=config.n_fft,
            hop_len=config.hop_length, win_type=config.window,
            win_len=config.win_length)
        self.sweep = RecRIRPIM(config.sample_rate, config.sweep_duration).sinesweep
        self.torch = torch

    def response_spectrum(self, rir):
        """Match Sweep v2: full waveform tail, n_fft zero guards, centered STFT."""
        torch = self.torch
        value = torch.as_tensor(np.asarray(rir), dtype=torch.float32)
        total = self.sweep.numel() + value.shape[-1] - 1
        n_fft = 1 << (total - 1).bit_length()
        response = torch.fft.irfft(
            torch.fft.rfft(self.sweep.float(), n=n_fft)
            * torch.fft.rfft(value, n=n_fft), n=n_fft)[..., :total]
        response = torch.nn.functional.pad(
            response, (self.config.n_fft, self.config.n_fft))
        return self.transforms.stft(response, "complex").numpy()

    def fit(self, direct, reverb):
        direct, reverb = normalize_rir_pair(
            direct, reverb, self.config.rir_length)
        return solve_full_convolution_ctf(
            self.response_spectrum(direct), self.response_spectrum(reverb),
            self.config.taps, self.config.regularization)


def pack_ctf(coefficients):
    """Pack causal complex [speaker, frequency, tap] as [tap, speaker, F, RI]."""
    value = np.asarray(coefficients)
    if value.ndim != 3 or not np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError("Expected finite complex [speaker, frequency, tap] CTFs")
    value = value.transpose(2, 0, 1)
    return np.stack((value.real, value.imag), axis=-1).astype(np.float32)
