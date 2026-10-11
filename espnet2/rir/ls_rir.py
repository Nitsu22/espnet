"""Causal, finite direct-to-reverb FIR references fitted to full sweep responses.

These are regularized identification references, not measured physical RIRs.
The FIR fit and the ordinary-sweep measurement are deliberately separate.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import fft, signal
from scipy.sparse.linalg import LinearOperator, cg

from espnet2.rir.oracle_ctf import normalize_rir_pair


@dataclass(frozen=True)
class LSRIRConfig:
    sample_rate: int = 16000
    rir_length: int = 32000
    filter_length: int = 32000
    sweep_duration: float = 8.192
    regularization: float = 1e-6
    solver_rtol: float = 1e-8
    max_iterations: int = 500

    def __post_init__(self):
        if (
            min(
                self.sample_rate,
                self.rir_length,
                self.filter_length,
                self.max_iterations,
            )
            < 1
            or not np.isfinite(self.sweep_duration)
            or self.sweep_duration <= 0
            or int(self.sample_rate * self.sweep_duration) < 256
            or not np.isfinite(self.regularization)
            or self.regularization <= 0
            or not np.isfinite(self.solver_rtol)
            or not 0 < self.solver_rtol < 1
        ):
            raise ValueError("Invalid LS RIR configuration; ridge must be positive")

    def as_dict(self):
        return asdict(self)


def _waveform(value):
    value = np.asarray(value, dtype=np.float64)
    if (
        value.ndim != 1
        or not value.size
        or not np.isfinite(value).all()
        or not np.any(value)
    ):
        raise ValueError("Expected a finite, nonzero mono waveform")
    return value


def solve_full_convolution_fir(
    x, y, taps, regularization=1e-6, rtol=1e-8, max_iterations=500
):
    """Minimize ||full_conv(x,g)-y||² + alpha ||x||² ||g||².

    g has exactly `taps` causal samples, starting at lag zero. Missing target
    samples, including the complete predicted tail, are zero. FFT matvecs
    implement the finite linear-convolution normal equations, not a circular
    spectral division followed by truncation. CG convergence is mandatory.
    """
    x, y = _waveform(x), _waveform(y)
    if (
        taps < 1
        or not np.isfinite(regularization)
        or regularization <= 0
        or not np.isfinite(rtol)
        or not 0 < rtol < 1
        or max_iterations < 1
    ):
        raise ValueError("Invalid finite FIR solver settings")
    frames = max(len(x) + taps - 1, len(y))
    nfft = fft.next_fast_len(frames, real=True)
    xf, yf = fft.rfft(x, n=nfft), fft.rfft(y, n=nfft)
    energy = float(np.dot(x, x))
    power = np.abs(xf) ** 2 / energy
    rhs = fft.irfft(xf.conj() * yf / energy, n=nfft)[:taps].copy()

    def multiply(value):
        result = fft.irfft(power * fft.rfft(value, n=nfft), n=nfft)[:taps]
        return result + regularization * value

    def precondition(value):
        # A principal submatrix of the positive-definite circulant inverse.
        # This accelerates CG; it does not change the finite FIR objective.
        return fft.irfft(fft.rfft(value, n=nfft) / (power + regularization), n=nfft)[
            :taps
        ].copy()

    operator = LinearOperator((taps, taps), matvec=multiply, dtype=np.float64)
    preconditioner = LinearOperator((taps, taps), matvec=precondition, dtype=np.float64)
    iterations = 0

    def count_iteration(_):
        nonlocal iterations
        iterations += 1

    coefficients, status = cg(
        operator,
        rhs,
        x0=precondition(rhs),
        M=preconditioner,
        rtol=rtol,
        atol=0,
        maxiter=max_iterations,
        callback=count_iteration,
    )
    normal_relative_residual = float(
        np.linalg.norm(multiply(coefficients) - rhs)
        / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    )
    if (
        status != 0
        or not np.isfinite(coefficients).all()
        or normal_relative_residual > rtol * 1.05
    ):
        raise RuntimeError(
            f"LS FIR did not converge: status={status}, iterations={iterations}, "
            f"normal_relative_residual={normal_relative_residual:.3g}"
        )
    prediction = signal.fftconvolve(x, coefficients)
    residual = np.pad(prediction, (0, frames - len(prediction))) - np.pad(
        y, (0, frames - len(y))
    )
    squared_error = float(np.dot(residual, residual))
    target_energy = float(np.dot(y, y))
    return coefficients, {
        "iterations": iterations,
        "normal_relative_residual": normal_relative_residual,
        "relative_response_mse": squared_error / target_energy,
        "ridge_absolute": regularization * energy,
        "objective": squared_error
        + regularization * energy * float(np.dot(coefficients, coefficients)),
        "response_samples": frames,
    }


def ordinary_sweep_measurement(coefficients, sweep, inverse, sample_rate, length):
    """Ordinary sweep*g followed by the ORIGINAL sweep inverse, as in PIM.

    Preserve PIM's peak-minus-2.5-ms crop and conditional peak normalization.
    Evaluation later uses the shared scorer's independent peak normalization.
    No direct-path RIR is applied here: doing that would restore the physical
    clean-to-reverb basis instead of the intended direct-to-reverb reference.
    """
    coefficients, sweep, inverse = map(_waveform, (coefficients, sweep, inverse))
    if sample_rate < 1 or length < 1:
        raise ValueError("Invalid measurement sample rate or output length")
    response = signal.fftconvolve(sweep, coefficients)
    recovered = signal.fftconvolve(inverse, response)
    peak = int(np.argmax(np.abs(recovered)))
    start = max(0, peak - int(sample_rate * 0.0025))
    recovered = recovered[start:]
    magnitude = float(np.abs(recovered).max())
    if magnitude > 1:
        recovered /= magnitude
    output = np.zeros(length, dtype=np.float64)
    output[: min(length, len(recovered))] = recovered[:length]
    if not np.isfinite(output).all() or not np.any(output):
        raise ValueError("Invalid measured LS RIR")
    return output, {
        "measurement_peak": peak,
        "measurement_crop_start": start,
        "measurement_pre_normalization_peak": magnitude,
    }


class LSRIRFitter:
    def __init__(self, config=LSRIRConfig()):
        from espnet2.rir.rec_rir.pim import RecRIRPIM

        self.config = config
        pim = RecRIRPIM(config.sample_rate, config.sweep_duration)
        # Use the exact existing PIM excitation/inverse, including its ramps
        # and 512-sample guards. Float64 arithmetic is used for the LS solve.
        self.sweep = pim.sinesweep.numpy().astype(np.float64)
        self.inverse = pim.invfilter.numpy().astype(np.float64)

    def fit(self, direct, reverb):
        direct, reverb = normalize_rir_pair(direct, reverb, self.config.rir_length)
        x = signal.fftconvolve(self.sweep, direct)
        y = signal.fftconvolve(self.sweep, reverb)
        coefficients, metrics = solve_full_convolution_fir(
            x,
            y,
            self.config.filter_length,
            self.config.regularization,
            self.config.solver_rtol,
            self.config.max_iterations,
        )
        physical_prediction = signal.fftconvolve(direct, coefficients)
        frames = max(len(physical_prediction), len(reverb))
        difference = np.pad(physical_prediction, (0, frames - len(physical_prediction)))
        difference -= np.pad(reverb, (0, frames - len(reverb)))
        metrics["relative_physical_pair_mse"] = float(
            np.dot(difference, difference) / np.dot(reverb, reverb)
        )
        reference, measurement = ordinary_sweep_measurement(
            coefficients,
            self.sweep,
            self.inverse,
            self.config.sample_rate,
            self.config.rir_length,
        )
        metrics.update(measurement)
        return coefficients, reference, metrics
