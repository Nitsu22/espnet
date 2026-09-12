# SPDX-License-Identifier: MIT
# Gated DeltaNet reference design adapted from Flash Linear Attention.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# See LICENSE.gated_deltanet for the upstream license.

"""Offline bidirectional Gated DeltaNet with a portable PyTorch recurrence.

The gated delta rule and layer design follow Yang et al. (ICLR 2025) and
fla-org/flash-linear-attention, commit 516143e31fce09925e6c39ac37148444bad176c4.
This is a reference backend, not FLA's optimized CUDA kernel. Every call starts
from zero state; neither recurrent state nor convolution state is cached.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def gated_delta_rule(q, k, v, log_decay, beta):
    """Apply the gated delta rule to [batch, time, heads, features] tensors.

    q and k must already be L2-normalized. Accumulate in FP32 (FP64 for
    double inputs). The decay is applied BEFORE computing the delta residual:
    S' = exp(g) S; S = S' + beta k (v - k^T S')^T; o = q^T S / sqrt(K).
    No T-by-T attention matrix is constructed.
    """
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, log_decay, beta = [
        x.to(dtype) for x in (q, k, v, log_decay, beta)
    ]
    batch, time, heads, key_dim = q.shape
    if time == 0:
        raise ValueError("Gated DeltaNet requires at least one frame")
    state = v.new_zeros(batch, heads, key_dim, v.shape[-1])
    outputs = []
    for t in range(time):
        key = k[:, t]
        state = log_decay[:, t].exp()[..., None, None] * state
        residual = v[:, t] - (key[..., None] * state).sum(dim=-2)
        update = beta[:, t, :, None] * residual
        state = state + key[..., None] * update[..., None, :]
        outputs.append((q[:, t, :, :, None] * state).sum(-2))
    return torch.stack(outputs, dim=1) / math.sqrt(key_dim)


def chunk_gated_delta_rule(q, k, v, log_decay, beta, chunk_size=32):
    """Exact chunkwise delta rule using batched triangular solves (Torch >=2.1).

    This is the same recurrence, without storing a K-by-V state per frame.
    A unit lower-triangular system resolves intra-chunk delta residuals; only
    chunk-boundary states are recurrent. Padding uses zero updates/zero decay.
    """
    if chunk_size <= 0 or q.shape[1] == 0:
        raise ValueError("chunk_size and sequence length must be positive")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, log_decay, beta = [
            t.to(dtype) for t in (q, k, v, log_decay, beta)
        ]
        batch, time, heads, key_dim = q.shape
        size = chunk_size
        padding = (-time) % size

        def split(t):
            t = F.pad(t.transpose(1, 2), (0, 0, 0, padding))
            return t.reshape(batch, heads, -1, size, t.shape[-1])

        q, k, v = [split(t) for t in (q, k, v)]
        g = split(log_decay.unsqueeze(-1)).squeeze(-1).cumsum(-1)
        beta = split(beta.unsqueeze(-1))
        # Mask BEFORE exponentiation, avoiding exp(large positive) above the
        # diagonal when cumulative negative decays have large magnitude.
        lower = torch.ones(size, size, device=q.device, dtype=torch.bool).tril()
        difference = g.unsqueeze(-1) - g.unsqueeze(-2)
        decay = difference.masked_fill(~lower, -torch.inf).exp()
        k_beta = k * beta
        system = torch.eye(size, device=q.device, dtype=dtype) + (
            (k_beta @ k.transpose(-1, -2)) * decay
        ).tril(-1)
        rhs = torch.cat((v * beta, k_beta * g.exp().unsqueeze(-1)), -1)
        solved = torch.linalg.solve_triangular(
            system, rhs, upper=False, unitriangular=True
        )
        values, keys = solved.split((v.shape[-1], key_dim), dim=-1)
        q = q / math.sqrt(key_dim)
        attention = (q @ k.transpose(-1, -2)) * decay
        state = v.new_zeros(batch, heads, key_dim, v.shape[-1])
        outputs = []
        for i in range(q.shape[2]):
            residual = values[:, :, i] - keys[:, :, i] @ state
            outputs.append(
                (q[:, :, i] * g[:, :, i].exp().unsqueeze(-1)) @ state
                + attention[:, :, i] @ residual
            )
            end_decay = g[:, :, i, -1]
            weighted_keys = k[:, :, i] * (
                end_decay.unsqueeze(-1) - g[:, :, i]
            ).exp().unsqueeze(-1)
            state = (state * end_decay.exp()[..., None, None]
                     + weighted_keys.transpose(-1, -2) @ residual)
        output = torch.stack(outputs, dim=2).flatten(2, 3)[:, :, :time]
        return output.transpose(1, 2)


class GatedDeltaNetDirection(nn.Module):
    """One causal direction with short Q/K/V convolutions and gated RMSNorm."""

    def __init__(
        self, dim, heads, head_dim, value_dim, conv_size, eps, backend="recurrent"
    ):
        super().__init__()
        if backend not in ("recurrent", "chunk"):
            raise ValueError(f"Unknown Gated DeltaNet backend: {backend}")
        self.backend = backend
        self.heads = heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.eps = eps
        self.conv_size = conv_size
        key_width, value_width = heads * head_dim, heads * value_dim
        self.q_proj = nn.Linear(dim, key_width, bias=False)
        self.k_proj = nn.Linear(dim, key_width, bias=False)
        self.v_proj = nn.Linear(dim, value_width, bias=False)
        self.q_conv = nn.Conv1d(
            key_width, key_width, conv_size, groups=key_width, bias=False
        )
        self.k_conv = nn.Conv1d(
            key_width, key_width, conv_size, groups=key_width, bias=False
        )
        self.v_conv = nn.Conv1d(
            value_width, value_width, conv_size, groups=value_width, bias=False
        )
        self.a_proj = nn.Linear(dim, heads, bias=False)
        self.b_proj = nn.Linear(dim, heads, bias=False)
        self.g_proj = nn.Linear(dim, value_width, bias=False)
        self.A_log = nn.Parameter(torch.empty(heads))
        self.dt_bias = nn.Parameter(torch.empty(heads))
        self.norm_weight = nn.Parameter(torch.ones(value_dim))
        self.o_proj = nn.Linear(value_width, dim, bias=False)
        self.espnet_initialization_fn()

    def espnet_initialization_fn(self):
        """Keep decay initialization valid after ESPnet-wide initialization."""
        with torch.no_grad():
            self.A_log.copy_(
                torch.empty_like(self.A_log).uniform_(0, 16).clamp_min(1e-6).log()
            )
            dt = (
                torch.empty_like(self.dt_bias)
                .uniform_(math.log(0.001), math.log(0.1))
                .exp()
            )
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            self.norm_weight.fill_(1)

    def _project_conv(self, x, projection, convolution, head_dim):
        x = projection(x).transpose(1, 2)
        x = convolution(F.pad(x, (self.conv_size - 1, 0))).transpose(1, 2)
        return F.silu(x).reshape(x.shape[0], x.shape[1], self.heads, head_dim)

    def forward(self, x):
        q = self._project_conv(x, self.q_proj, self.q_conv, self.head_dim)
        k = self._project_conv(x, self.k_proj, self.k_conv, self.head_dim)
        v = self._project_conv(x, self.v_proj, self.v_conv, self.value_dim)
        # Gate exponentials, normalization, and recurrent state stay in FP32
        # even when projections are run under autocast.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        q, k = [F.normalize(t.to(dtype), dim=-1, eps=1e-6) for t in (q, k)]
        log_decay = -self.A_log.to(dtype).exp() * F.softplus(
            self.a_proj(x).to(dtype) + self.dt_bias.to(dtype)
        )
        beta = self.b_proj(x).to(dtype).sigmoid()
        rule = chunk_gated_delta_rule if self.backend == "chunk" else gated_delta_rule
        y = rule(q, k, v, log_decay, beta)
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        gate = self.g_proj(x).reshape_as(y)
        y = y * self.norm_weight.to(dtype) * F.silu(gate.to(dtype))
        return self.o_proj(y.flatten(-2).to(v.dtype))


class BidirectionalGatedDeltaNet(nn.Module):
    """Independent forward/backward GDN layers, concatenated then projected.

    Reversal happens BEFORE the backward layer's short causal convolutions;
    its output is reversed back before fusion. Input must be unpadded.
    """

    def __init__(
        self, dim, heads=4, head_dim=32, value_dim=64, conv_size=4,
        dropout=0.0, eps=1e-5, backend="recurrent",
    ):
        super().__init__()
        if min(dim, heads, head_dim, value_dim, conv_size) <= 0:
            raise ValueError("Gated DeltaNet dimensions and conv_size must be positive")
        self.forward_net = GatedDeltaNetDirection(
            dim, heads, head_dim, value_dim, conv_size, eps, backend
        )
        self.backward_net = GatedDeltaNetDirection(
            dim, heads, head_dim, value_dim, conv_size, eps, backend
        )
        self.fusion = nn.Linear(2 * dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        forward = self.forward_net(x)
        backward = self.backward_net(x.flip(1)).flip(1)
        return self.dropout(self.fusion(torch.cat((forward, backward), dim=-1)))
