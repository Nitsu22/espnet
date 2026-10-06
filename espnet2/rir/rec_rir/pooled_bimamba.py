"""Time BiMamba -> two-slot pooling -> shared frequency refinement for CTFs.

Lengths are intentionally not masked, matching the current TF-Locoformer
Sweep v2 baseline. The returned taps use that baseline's convolution ordering.
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding

from espnet2.rir.rec_rir.model import _build_mamba


class TimeBiMamba(nn.Module):
    def __init__(self, dim, d_state, d_conv, eps):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        specification = f"mamba({d_state},{d_conv})"
        self.forward_mamba = _build_mamba(dim, specification)
        self.backward_mamba = _build_mamba(dim, specification)
        self.projection = nn.Linear(2 * dim, dim)

    def forward(self, x):
        batch, frames, freqs, dim = x.shape
        sequence = self.norm(x).permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
        forward = self.forward_mamba(sequence.contiguous())
        backward = self.backward_mamba(sequence.flip(1).contiguous()).flip(1)
        update = self.projection(torch.cat((forward, backward), -1))
        return x + update.reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)


class LightFrequencyBlock(nn.Module):
    def __init__(self, dim, freqs, bottleneck, rank, kernel, eps):
        super().__init__()
        self.local_norm = nn.LayerNorm(dim, eps=eps)
        self.local = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.global_norm = nn.LayerNorm(dim, eps=eps)
        self.compress = nn.Linear(dim, bottleneck)
        # One frequency MLP shared by all compressed feature channels.
        self.frequency_mlp = nn.Sequential(nn.Linear(freqs, rank), nn.GELU(), nn.Linear(rank, freqs))
        self.expand = nn.Linear(bottleneck, dim)

    def forward(self, x):
        batch, frames, freqs, dim = x.shape
        sequence = x.reshape(batch * frames, freqs, dim)
        update = self.local(self.local_norm(sequence).transpose(1, 2)).transpose(1, 2)
        sequence = sequence + F.gelu(update)
        update = self.compress(self.global_norm(sequence)).transpose(1, 2)
        update = self.frequency_mlp(update).transpose(1, 2)
        return (sequence + self.expand(update)).reshape(batch, frames, freqs, dim)


class FrequencyAttentionBlock(nn.Module):
    def __init__(self, dim, heads, ffn_dim, kernel, eps):
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError("RoPE requires an even head dimension and dim divisible by heads")
        self.heads = heads
        self.norm_attention = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.rope = RotaryEmbedding(dim // heads)
        self.projection = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim, eps=eps)
        self.ffn_in = nn.Linear(dim, 2 * ffn_dim)
        self.local = nn.Conv1d(ffn_dim, ffn_dim, kernel, padding=kernel // 2, groups=ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, dim)

    def forward(self, x):
        batch, freqs, dim = x.shape
        qkv = self.qkv(self.norm_attention(x)).reshape(batch, freqs, 3, self.heads, dim // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        update = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        x = x + self.projection(update.transpose(1, 2).reshape(batch, freqs, dim))
        update = F.glu(self.ffn_in(self.norm_ffn(x)), dim=-1)
        update = F.silu(self.local(update.transpose(1, 2)).transpose(1, 2))
        return x + self.ffn_out(update)


class PooledBiMambaCTFPredictor(nn.Module):
    def __init__(self, input_dim=257, num_spk=2, ctf_taps=60, emb_dim=64,
                 pre_layers=4, post_layers=2, d_state=16, d_conv=4,
                 freq_bottleneck=16, freq_rank=16, local_kernel=5,
                 pooling_hidden=32, n_heads=4, ffn_dim=128,
                 head_dim=128, slot_embedding_std=0.02, eps=1e-5):
        super().__init__()
        positive = (input_dim, num_spk, ctf_taps, emb_dim, pre_layers, d_state,
                    d_conv, freq_bottleneck, freq_rank, local_kernel, pooling_hidden,
                    n_heads, ffn_dim, head_dim)
        if min(positive) <= 0 or post_layers < 0 or local_kernel % 2 != 1:
            raise ValueError("Use positive dimensions, nonnegative post_layers and an odd local kernel")
        if num_spk != 2:
            raise ValueError("The sweep PIT wrapper requires two speakers")
        self.input_dim = input_dim
        self._num_spk = num_spk
        self.ctf_taps = ctf_taps
        self.encoder = nn.Conv2d(2, emb_dim, kernel_size=3, padding=1)
        self.time_blocks = nn.ModuleList(TimeBiMamba(emb_dim, d_state, d_conv, eps) for _ in range(pre_layers))
        self.frequency_blocks = nn.ModuleList(
            LightFrequencyBlock(emb_dim, input_dim, freq_bottleneck, freq_rank, local_kernel, eps)
            for _ in range(pre_layers))
        self.pooling = nn.Sequential(nn.Linear(emb_dim, pooling_hidden), nn.GELU(),
                                     nn.Linear(pooling_hidden, num_spk * emb_dim))
        self.slot_embeddings = nn.Parameter(torch.empty(num_spk, emb_dim))
        nn.init.normal_(self.slot_embeddings, std=slot_embedding_std)
        self.post_blocks = nn.ModuleList(
            FrequencyAttentionBlock(emb_dim, n_heads, ffn_dim, local_kernel, eps)
            for _ in range(post_layers))
        self.head = nn.Sequential(nn.LayerNorm(emb_dim, eps=eps), nn.Linear(emb_dim, head_dim),
                                  nn.SiLU(), nn.Linear(head_dim, 2 * ctf_taps))

    def forward(self, input, ilens=None, additional=None):
        if input.ndim != 3 or not torch.is_complex(input) or input.shape[-1] != self.input_dim:
            raise ValueError("Expected complex STFT [B, T, input_dim]")
        x = self.encoder(torch.stack((input.real, input.imag), 1).float()).permute(0, 2, 3, 1)
        for time_block, frequency_block in zip(self.time_blocks, self.frequency_blocks):
            x = frequency_block(time_block(x))
        batch, frames, freqs, dim = x.shape
        scores = self.pooling(x).reshape(batch, frames, freqs, self.num_spk, dim)
        # Match baseline padding behavior: softmax over the full batch time axis.
        weights = scores.softmax(dim=1)
        pooled = (weights * x.unsqueeze(3)).sum(1).permute(0, 2, 1, 3)
        pooled = pooled + self.slot_embeddings[None, :, None, :]
        pooled = pooled.reshape(batch * self.num_spk, freqs, dim)
        for block in self.post_blocks:
            pooled = block(pooled)
        output = self.head(pooled).reshape(batch, self.num_spk, freqs, self.ctf_taps, 2).float()
        ctf = torch.complex(output[..., 0], output[..., 1]).contiguous()
        return ctf, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self._num_spk
