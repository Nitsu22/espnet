"""Differentiable, finite-length complex CTF convolution and inverse filtering.

All spectra use ``[batch, speaker, frequency, frame]`` and filters use
``[batch, speaker, frequency, tap]``.  A tap at index ``l`` multiplies the
input at ``t - l``.  FFTs implement zero-padded *linear* convolution, with
only the first ``T`` observation frames retained.
"""

import math
from numbers import Integral, Real
from typing import Optional

import torch
from torch import Tensor


def _check_signal(signal: Tensor, ctf: Tensor) -> None:
    if not isinstance(signal, Tensor) or not isinstance(ctf, Tensor):
        raise TypeError("signal and ctf must be torch tensors")
    if signal.ndim != 4 or ctf.ndim != 4:
        raise ValueError("signal and ctf must have four dimensions")
    if any(size == 0 for size in signal.shape + ctf.shape):
        raise ValueError("signal and ctf dimensions must be nonempty")
    if signal.shape[:3] != ctf.shape[:3]:
        raise ValueError("signal and ctf batch, speaker, and frequency must match")
    if signal.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("signal must have dtype complex64 or complex128")
    if ctf.dtype != signal.dtype or ctf.device != signal.device:
        raise ValueError("signal and ctf must have the same dtype and device")
    if not torch.isfinite(signal).all() or not torch.isfinite(ctf).all():
        raise ValueError("signal and ctf must be finite, including padding")


def _frame_mask(signal: Tensor, frame_lengths: Optional[Tensor]) -> Tensor:
    batch, speakers, _, frames = signal.shape
    if frame_lengths is None:
        return torch.ones(
            (batch, 1, 1, frames), dtype=torch.bool, device=signal.device
        )
    if not isinstance(frame_lengths, Tensor):
        raise TypeError("frame_lengths must be an integer torch tensor")
    if frame_lengths.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("frame_lengths must have integer dtype")
    if frame_lengths.shape == (batch,):
        lengths = frame_lengths[:, None, None, None]
    elif frame_lengths.shape == (batch, speakers):
        lengths = frame_lengths[:, :, None, None]
    else:
        raise ValueError("frame_lengths must have shape [B] or [B, N]")
    if (lengths < 0).any() or (lengths > frames).any():
        raise ValueError("frame_lengths must lie between zero and T")
    lengths = lengths.to(device=signal.device)
    return torch.arange(frames, device=signal.device) < lengths


def _observation_weights(signal: Tensor, mask: Optional[Tensor]) -> Optional[Tensor]:
    if mask is None:
        return None
    if not isinstance(mask, Tensor):
        raise TypeError("observation_mask must be a torch tensor")
    if mask.dtype != torch.bool and not torch.is_floating_point(mask):
        raise TypeError("observation_mask must be boolean or real floating point")
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("observation_mask must be finite and nonnegative")
    batch, speakers, _, frames = signal.shape
    # Permit the common time-mask layouts without requiring callers to add axes.
    if mask.ndim == 2 and mask.shape[0] in (1, batch):
        mask = mask[:, None, None, :]
    elif (
        mask.ndim == 3
        and mask.shape[0] in (1, batch)
        and mask.shape[1] in (1, speakers)
    ):
        mask = mask[:, :, None, :]
    try:
        mask = torch.broadcast_to(mask, signal.shape)
    except RuntimeError as error:
        raise ValueError("observation_mask must broadcast to [B, N, F, T]") from error
    return mask.to(device=signal.device, dtype=signal.real.dtype)


class _FFTCTF:
    """Cache the filter transform for the forward and adjoint CG operations."""

    def __init__(self, ctf: Tensor, frames: int, valid: Tensor):
        self.frames = frames
        self.valid = valid
        self.fft_size = 1 << (frames + ctf.shape[-1] - 2).bit_length()
        self.filter_fft = torch.fft.fft(ctf, n=self.fft_size, dim=-1)

    def forward(self, signal: Tensor) -> Tensor:
        signal = signal.masked_fill(~self.valid, 0)
        spectrum = torch.fft.fft(signal, n=self.fft_size, dim=-1)
        result = torch.fft.ifft(spectrum * self.filter_fft, dim=-1)
        return result[..., : self.frames].masked_fill(~self.valid, 0)

    def adjoint(self, observation: Tensor) -> Tensor:
        observation = observation.masked_fill(~self.valid, 0)
        spectrum = torch.fft.fft(observation, n=self.fft_size, dim=-1)
        result = torch.fft.ifft(spectrum * self.filter_fft.conj(), dim=-1)
        return result[..., : self.frames].masked_fill(~self.valid, 0)


def ctf_forward(
    signal: Tensor,
    ctf: Tensor,
    frame_lengths: Optional[Tensor] = None,
    observation_mask: Optional[Tensor] = None,
) -> Tensor:
    """Apply causal linear CTF convolution, truncated to the input frame count.

    ``frame_lengths`` has shape ``[B]`` or ``[B, N]``.  Both input and output
    padding are zero, so arbitrary finite padded inputs cannot leak into valid
    frames.  ``observation_mask`` is a boolean mask or nonnegative real weight
    broadcastable to the output; ``[B, T]`` and ``[B, N, T]`` also work.  It
    multiplies the observed output, and does not remove preceding input context.
    """
    _check_signal(signal, ctf)
    valid = _frame_mask(signal, frame_lengths)
    weights = _observation_weights(signal, observation_mask)
    result = _FFTCTF(ctf, signal.shape[-1], valid).forward(signal)
    return result if weights is None else result * weights


def ctf_adjoint(
    observation: Tensor,
    ctf: Tensor,
    frame_lengths: Optional[Tensor] = None,
    observation_mask: Optional[Tensor] = None,
) -> Tensor:
    """Apply the exact complex adjoint of :func:`ctf_forward`.

    The real observation weights are applied *before* the conjugate correlation.
    Thus this is also the adjoint for nonbinary masks, not merely for selections.
    """
    _check_signal(observation, ctf)
    valid = _frame_mask(observation, frame_lengths)
    weights = _observation_weights(observation, observation_mask)
    if weights is not None:
        observation = observation * weights
    return _FFTCTF(ctf, observation.shape[-1], valid).adjoint(observation)


def regularized_deconvolution(
    observation: Tensor,
    ctf: Tensor,
    prior: Tensor,
    frame_lengths: Optional[Tensor] = None,
    observation_mask: Optional[Tensor] = None,
    gamma: float = 0.1,
    eps: float = 1e-6,
    iterations: int = 8,
) -> Tensor:
    """Solve ``(A^H W A + lambda I)s = A^H W y + lambda s0`` with CG.

    Here ``A`` is the unweighted, finite causal convolution and
    ``lambda = gamma * sum(abs(ctf)**2) + eps`` for each speaker/frequency.
    The observation mask is ``W`` and is applied exactly once, including when
    its values are nonbinary.  All ``iterations`` are differentiable unrolled
    CG steps, beginning at ``prior``.  Directions that have converged are frozen
    using boolean masks to avoid division by zero.  The result retains the
    complex64 or complex128 input dtype and has zero padding.
    """
    _check_signal(observation, ctf)
    if not isinstance(prior, Tensor):
        raise TypeError("prior must be a torch tensor")
    if (
        prior.shape != observation.shape
        or prior.dtype != observation.dtype
        or prior.device != observation.device
    ):
        raise ValueError(
            "prior and observation must have the same shape, dtype, device"
        )
    if not torch.isfinite(prior).all():
        raise ValueError("prior must be finite, including padding")
    for name, value in (("gamma", gamma), ("eps", eps)):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if gamma < 0 or eps <= 0:
        raise ValueError("gamma must be nonnegative and eps must be positive")
    if isinstance(iterations, bool) or not isinstance(iterations, Integral):
        raise TypeError("iterations must be a positive integer")
    if iterations <= 0:
        raise ValueError("iterations must be a positive integer")

    valid = _frame_mask(observation, frame_lengths)
    weights = _observation_weights(observation, observation_mask)
    operator = _FFTCTF(ctf, observation.shape[-1], valid)
    energy = (ctf.real.square() + ctf.imag.square()).sum(dim=-1, keepdim=True)
    regularization = gamma * energy + eps
    # Rescale each independent system.  This leaves the solution unchanged and
    # avoids tiny CG denominators when both the filter and regularizer are small.
    system_scale = energy + regularization
    if (
        not torch.isfinite(system_scale).all()
        or not torch.isfinite(regularization).all()
        or (regularization <= 0).any()
    ):
        raise ValueError("ctf energy and regularization must be finite and positive")
    regularization_scaled = regularization / system_scale

    def weighted_adjoint(value: Tensor) -> Tensor:
        if weights is not None:
            value = value * weights
        return operator.adjoint(value)

    def normal(value: Tensor) -> Tensor:
        return (
            weighted_adjoint(operator.forward(value)) / system_scale
            + regularization_scaled * value
        )

    estimate = prior.masked_fill(~valid, 0)
    # Algebraically b - M*s0; this form avoids cancellation of lambda*s0.
    residual = weighted_adjoint(observation - operator.forward(estimate)) / system_scale
    # CG is homogeneous in its right-hand side.  Solve a normalized correction
    # so squaring a nearly silent residual cannot underflow and huge residuals
    # cannot overflow.  This scaling remains in the autograd graph.
    residual_scale = residual.abs().amax(dim=-1, keepdim=True)
    safe_scale = torch.where(
        residual_scale > 0, residual_scale, torch.ones_like(residual_scale)
    )
    residual = residual / safe_scale
    correction = torch.zeros_like(estimate)
    direction = residual
    residual_norm = residual.abs().square().sum(dim=-1, keepdim=True)
    threshold = residual_norm * (8 * torch.finfo(observation.real.dtype).eps) ** 2
    active = residual_norm > 0
    for _ in range(iterations):
        normal_direction = normal(direction)
        denominator = (direction.conj() * normal_direction).real.sum(
            dim=-1, keepdim=True
        )
        step_active = active & (denominator > 0)
        safe_denominator = torch.where(
            step_active, denominator, torch.ones_like(denominator)
        )
        alpha = torch.where(step_active, residual_norm / safe_denominator, 0)
        correction = correction + alpha * direction
        next_residual = residual - alpha * normal_direction
        next_norm = next_residual.abs().square().sum(dim=-1, keepdim=True)
        next_active = step_active & (next_norm > threshold)
        safe_norm = torch.where(
            next_active, residual_norm, torch.ones_like(residual_norm)
        )
        beta = torch.where(next_active, next_norm / safe_norm, 0)
        direction = (next_residual + beta * direction).masked_fill(~next_active, 0)
        residual = next_residual
        residual_norm = next_norm
        active = next_active
    return (estimate + residual_scale * correction).masked_fill(~valid, 0)
