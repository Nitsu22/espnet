"""SPMamba-inspired local unfolding and full-frame attention for CTFs.

This lightweight adaptation uses Mamba-1 expand=2 and channel LayerNorm,
retains the pooled CTF head, and is not a reproduction of the speech separator.
Unfold never subsamples time or frequency: kernel is odd, stride is one.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from espnet2.rir.rec_rir.model import _build_mamba


class RMSNorm(nn.Module):
    """RMSNorm compatible with the existing PyTorch 2.1 runtime."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class UnfoldBiMamba(nn.Module):
    """[N, length, D] -> channel-major local patches -> BiMamba -> deconv."""

    def __init__(self, dim, kernel, d_state, d_conv, eps):
        super().__init__()
        if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1 or kernel % 2 != 1:
            raise ValueError("unfold_kernel must be a positive odd integer")
        self.dim = dim
        self.kernel = kernel
        self.norm = nn.LayerNorm(dim, eps=eps)
        patch_dim = dim * kernel
        self.forward_norm = RMSNorm(patch_dim, eps)
        self.backward_norm = RMSNorm(patch_dim, eps)
        specification = f"mamba({d_state},{d_conv})"
        self.forward_mamba = _build_mamba(patch_dim, specification)
        self.backward_mamba = _build_mamba(patch_dim, specification)
        self.restore = nn.ConvTranspose1d(2 * patch_dim, dim, kernel,
                                          stride=1, padding=kernel // 2)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("Expected sequences [N, length, D]")
        # F.unfold follows channel-major [D, kernel] patch ordering, as in GridNet.
        patches = F.unfold(self.norm(x).transpose(1, 2).unsqueeze(-1),
                           kernel_size=(self.kernel, 1),
                           padding=(self.kernel // 2, 0)).transpose(1, 2)
        forward = self.forward_mamba(self.forward_norm(patches).contiguous()) + patches
        reverse = patches.flip(1)
        backward = (self.backward_mamba(self.backward_norm(reverse).contiguous()) + reverse).flip(1)
        update = self.restore(torch.cat((forward, backward), -1).transpose(1, 2)).transpose(1, 2)
        return x + update


class UnfoldAxisBiMamba(nn.Module):
    """Apply independent frequency or time sequences without mixing their batches."""

    def __init__(self, dim, kernel, d_state, d_conv, eps, axis):
        super().__init__()
        if axis not in ("time", "frequency"):
            raise ValueError("axis must be time or frequency")
        self.axis = axis
        self.sequence = UnfoldBiMamba(dim, kernel, d_state, d_conv, eps)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected [B, T, F, D]")
        batch, frames, freqs, dim = x.shape
        if self.axis == "frequency":
            return self.sequence(x.reshape(batch * frames, freqs, dim)).reshape(batch, frames, freqs, dim)
        sequence = x.permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
        return self.sequence(sequence).reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)


class FrameAttention(nn.Module):
    """SPMamba-style attention between whole frames, preserving all value bins.

Queries/keys concatenate frequency bins within each head. Values retain the
full frequency-feature grid; attention is over T, never over speaker slots.
Padding handling matches the existing Sweep v2 architecture comparison.
"""

    def __init__(self, dim, heads, qk_channels, eps):
        super().__init__()
        if dim % heads or isinstance(qk_channels, bool) or not isinstance(qk_channels, int) or qk_channels < 1:
            raise ValueError("Use heads dividing dim and positive integer qk_channels")
        self.heads = heads
        self.qk_channels = qk_channels
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 2 * heads * qk_channels + dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        batch, frames, freqs, dim = x.shape
        width = self.heads * self.qk_channels
        q, k, v = self.qkv(self.norm(x)).split((width, width, dim), dim=-1)
        def layout(y, channels):
            return y.reshape(batch, frames, freqs, self.heads, channels).permute(0, 3, 1, 2, 4).flatten(3)
        q = layout(q, self.qk_channels)
        k = layout(k, self.qk_channels)
        v = layout(v, dim // self.heads)
        weights = ((q @ k.transpose(-1, -2)) / math.sqrt(freqs * self.qk_channels)).softmax(-1)
        update = (weights @ v).reshape(batch, self.heads, frames, freqs, dim // self.heads)
        update = update.permute(0, 2, 3, 1, 4).reshape(batch, frames, freqs, dim)
        return x + self.projection(update)
