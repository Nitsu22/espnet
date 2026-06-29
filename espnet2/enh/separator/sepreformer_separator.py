# Copyright 2026 Daichi Nitsu
#
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.enh.separator.abs_separator import AbsSeparator


class LayerScale(nn.Module):
    def __init__(self, dims: int, input_size: int, layer_scale_init: float = 1.0e-5):
        super().__init__()
        if dims == 1:
            shape = (input_size,)
        elif dims == 2:
            shape = (1, input_size)
        elif dims == 3:
            shape = (1, 1, input_size)
        else:
            raise ValueError(f"Unsupported dims: {dims}")

        self.layer_scale = nn.Parameter(
            torch.ones(shape) * layer_scale_init, requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layer_scale


class Masking(nn.Module):
    def __init__(
        self,
        input_dim: int,
        activation_mask: str = "Sigmoid",
        concat_opt: Optional[bool] = None,
    ):
        super().__init__()
        self.concat_opt = concat_opt
        if self.concat_opt:
            self.pw_conv = nn.Conv1d(input_dim * 2, input_dim, 1, stride=1, padding=0)

        if activation_mask == "Sigmoid":
            self.gate_act = nn.Sigmoid()
        elif activation_mask == "ReLU":
            self.gate_act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation_mask: {activation_mask}")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.concat_opt:
            x = torch.cat([x, skip], dim=-2)
            x = self.pw_conv(x)
        return self.gate_act(x) * skip


class GCFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout_rate: float,
        layer_scale_init: float = 1.0e-5,
    ):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 6),
        )
        self.depthwise = nn.Conv1d(
            in_channels * 6,
            in_channels * 6,
            3,
            padding=1,
            groups=in_channels * 6,
        )
        self.net2 = nn.Sequential(
            nn.GLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels * 3, in_channels),
            nn.Dropout(dropout_rate),
        )
        self.layer_scale = LayerScale(
            dims=3, input_size=in_channels, layer_scale_init=layer_scale_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net1(x)
        y = y.permute(0, 2, 1).contiguous()
        y = self.depthwise(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.net2(y)
        return x + self.layer_scale(y)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        in_channels: int,
        dropout_rate: float,
        layer_scale_init: float = 1.0e-5,
    ):
        super().__init__()
        assert in_channels % n_head == 0, (in_channels, n_head)
        self.d_k = in_channels // n_head
        self.h = n_head
        self.layer_norm = nn.LayerNorm(in_channels)
        self.linear_q = nn.Linear(in_channels, in_channels)
        self.linear_k = nn.Linear(in_channels, in_channels)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.linear_out = nn.Linear(in_channels, in_channels)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_scale = LayerScale(
            dims=3, input_size=in_channels, layer_scale_init=layer_scale_init
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_k: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        if pos_k is not None:
            reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k)
            reshape_q = reshape_q.transpose(0, 1)
            rel_scores = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            rel_scores = rel_scores.transpose(0, 1)
            rel_scores = rel_scores.view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = scores + rel_scores
        scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        x = self.dropout(self.linear_out(x))
        return self.layer_scale(x)


class EGA(nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = nn.ModuleDict(
            {
                "self_attn": MultiHeadAttention(
                    n_head=num_mha_heads,
                    in_channels=in_channels,
                    dropout_rate=dropout_rate,
                ),
                "linear": nn.Sequential(
                    nn.LayerNorm(normalized_shape=in_channels),
                    nn.Linear(in_features=in_channels, out_features=in_channels),
                    nn.Sigmoid(),
                ),
            }
        )

    def forward(self, x: torch.Tensor, pos_k: torch.Tensor) -> torch.Tensor:
        down_len = pos_k.shape[0]
        x_down = F.adaptive_avg_pool1d(input=x, output_size=down_len)
        x = x.permute(0, 2, 1)
        x_down = x_down.permute(0, 2, 1)
        x_down = self.block["self_attn"](x_down, pos_k, None)
        x_down = x_down.permute(0, 2, 1)
        x_downup = F.interpolate(input=x_down, size=x.shape[1], mode="nearest")
        x_downup = x_downup.permute(0, 2, 1)
        return x + self.block["linear"](x) * x_downup


class CLA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        dropout_rate: float,
        layer_scale_init: float = 1.0e-5,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)
        self.linear1 = nn.Linear(in_channels, in_channels * 2)
        self.glu = nn.GLU()
        self.dw_conv_1d = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            groups=in_channels,
        )
        self.linear2 = nn.Linear(in_channels, 2 * in_channels)
        self.bn = nn.BatchNorm1d(2 * in_channels)
        self.linear3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(2 * in_channels, in_channels),
            nn.Dropout(dropout_rate),
        )
        self.layer_scale = LayerScale(
            dims=3, input_size=in_channels, layer_scale_init=layer_scale_init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layer_norm(x)
        y = self.linear1(y)
        y = self.glu(y)
        y = y.permute(0, 2, 1)
        y = self.dw_conv_1d(y)
        y = y.permute(0, 2, 1)
        y = self.linear2(y)
        y = y.permute(0, 2, 1)
        y = self.bn(y)
        y = y.permute(0, 2, 1)
        y = self.linear3(y)
        return x + self.layer_scale(y)


class GlobalBlock(nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.block = nn.ModuleDict(
            {
                "ega": EGA(
                    num_mha_heads=num_mha_heads,
                    in_channels=in_channels,
                    dropout_rate=dropout_rate,
                ),
                "gcfn": GCFN(in_channels=in_channels, dropout_rate=dropout_rate),
            }
        )

    def forward(self, x: torch.Tensor, pos_k: torch.Tensor) -> torch.Tensor:
        x = self.block["ega"](x, pos_k)
        x = self.block["gcfn"](x)
        return x.permute(0, 2, 1)


class LocalBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.block = nn.ModuleDict(
            {
                "cla": CLA(in_channels, kernel_size, dropout_rate),
                "gcfn": GCFN(in_channels, dropout_rate),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block["cla"](x)
        return self.block["gcfn"](x)


class SpkAttention(nn.Module):
    def __init__(self, in_channels: int, num_mha_heads: int, dropout_rate: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            n_head=num_mha_heads,
            in_channels=in_channels,
            dropout_rate=dropout_rate,
        )
        self.feed_forward = GCFN(in_channels=in_channels, dropout_rate=dropout_rate)

    def forward(self, x: torch.Tensor, num_spk: int) -> torch.Tensor:
        batch, feat_dim, frames = x.shape
        x = x.view(batch // num_spk, num_spk, feat_dim, frames).contiguous()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, num_spk, feat_dim).contiguous()
        x = x + self.self_attn(x, None, None)
        x = x.view(batch // num_spk, frames, num_spk, feat_dim).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch, feat_dim, frames).contiguous()
        x = x.permute(0, 2, 1)
        x = self.feed_forward(x)
        return x.permute(0, 2, 1)


class AudioEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        bias: bool,
    ):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise RuntimeError("AudioEncoder only accepts 1D, 2D, or 3D tensors.")
        return self.gelu(self.conv1d(x))


class FeatureProjector(nn.Module):
    def __init__(
        self,
        num_channels: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=num_channels, eps=1.0e-8)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d(self.norm(x))


class RelativePositionalEncoding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        maxlen: int,
        embed_v: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.embedding_dim = self.in_channels // self.num_heads
        self.maxlen = maxlen
        self.pe_k = nn.Embedding(
            num_embeddings=2 * maxlen,
            embedding_dim=self.embedding_dim,
        )
        self.pe_v = (
            nn.Embedding(
                num_embeddings=2 * maxlen,
                embedding_dim=self.embedding_dim,
            )
            if embed_v
            else None
        )

    def forward(
        self, pos_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pos_seq = pos_seq.clamp(-self.maxlen, self.maxlen - 1) + self.maxlen
        pe_k_output = self.pe_k(pos_seq)
        pe_v_output = self.pe_v(pos_seq) if self.pe_v is not None else None
        return pe_k_output, pe_v_output


class DownConvLayer(nn.Module):
    def __init__(self, in_channels: int, samp_kernel_size: int):
        super().__init__()
        self.down_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=samp_kernel_size,
            stride=2,
            padding=(samp_kernel_size - 1) // 2,
            groups=in_channels,
        )
        self.bn = nn.BatchNorm1d(num_features=in_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.down_conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x.permute(0, 2, 1)


class SepEncStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_mha_heads: int,
        global_dropout_rate: float,
        local_kernel_size: int,
        local_dropout_rate: float,
        samp_kernel_size: int,
        down_conv: bool = True,
    ):
        super().__init__()
        self.g_block_1 = GlobalBlock(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=global_dropout_rate,
        )
        self.l_block_1 = LocalBlock(
            in_channels=in_channels,
            kernel_size=local_kernel_size,
            dropout_rate=local_dropout_rate,
        )
        self.g_block_2 = GlobalBlock(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=global_dropout_rate,
        )
        self.l_block_2 = LocalBlock(
            in_channels=in_channels,
            kernel_size=local_kernel_size,
            dropout_rate=local_dropout_rate,
        )
        self.downconv = (
            DownConvLayer(
                in_channels=in_channels,
                samp_kernel_size=samp_kernel_size,
            )
            if down_conv
            else None
        )

    def forward(
        self, x: torch.Tensor, pos_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.g_block_1(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_1(x)
        x = x.permute(0, 2, 1).contiguous()

        x = self.g_block_2(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_2(x)
        x = x.permute(0, 2, 1).contiguous()

        skip = x
        if self.downconv is not None:
            x = x.permute(0, 2, 1).contiguous()
            x = self.downconv(x)
            x = x.permute(0, 2, 1).contiguous()
        return x, skip


class SpkSplitStage(nn.Module):
    def __init__(self, in_channels: int, num_spk: int):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Conv1d(in_channels, 4 * in_channels * num_spk, kernel_size=1),
            nn.GLU(dim=-2),
            nn.Conv1d(2 * in_channels * num_spk, in_channels * num_spk, kernel_size=1),
        )
        self.norm = nn.GroupNorm(1, in_channels, eps=1.0e-8)
        self.num_spk = num_spk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        batch, _, frames = x.shape
        x = x.view(batch * self.num_spk, -1, frames).contiguous()
        return self.norm(x)


class SepDecStage(nn.Module):
    def __init__(
        self,
        num_spk: int,
        in_channels: int,
        num_mha_heads: int,
        global_dropout_rate: float,
        local_kernel_size: int,
        local_dropout_rate: float,
        spk_attention_dropout_rate: float,
    ):
        super().__init__()
        self.g_block_1 = GlobalBlock(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=global_dropout_rate,
        )
        self.l_block_1 = LocalBlock(
            in_channels=in_channels,
            kernel_size=local_kernel_size,
            dropout_rate=local_dropout_rate,
        )
        self.spk_attn_1 = SpkAttention(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=spk_attention_dropout_rate,
        )

        self.g_block_2 = GlobalBlock(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=global_dropout_rate,
        )
        self.l_block_2 = LocalBlock(
            in_channels=in_channels,
            kernel_size=local_kernel_size,
            dropout_rate=local_dropout_rate,
        )
        self.spk_attn_2 = SpkAttention(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=spk_attention_dropout_rate,
        )

        self.g_block_3 = GlobalBlock(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=global_dropout_rate,
        )
        self.l_block_3 = LocalBlock(
            in_channels=in_channels,
            kernel_size=local_kernel_size,
            dropout_rate=local_dropout_rate,
        )
        self.spk_attn_3 = SpkAttention(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            dropout_rate=spk_attention_dropout_rate,
        )
        self.num_spk = num_spk

    def forward(
        self, x: torch.Tensor, pos_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.g_block_1(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.spk_attn_1(x, self.num_spk)

        x = self.g_block_2(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.spk_attn_2(x, self.num_spk)

        x = self.g_block_3(x, pos_k)
        x = x.permute(0, 2, 1).contiguous()
        x = self.l_block_3(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.spk_attn_3(x, self.num_spk)
        return x, x


class Separator(nn.Module):
    def __init__(
        self,
        num_stages: int,
        num_spk: int,
        in_channels: int,
        num_mha_heads: int,
        maxlen: int,
        embed_v: bool,
        global_dropout_rate: float,
        local_kernel_size: int,
        local_dropout_rate: float,
        samp_kernel_size: int,
        spk_attention_dropout_rate: float,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.pos_emb = RelativePositionalEncoding(
            in_channels=in_channels,
            num_heads=num_mha_heads,
            maxlen=maxlen,
            embed_v=embed_v,
        )

        self.enc_stages = nn.ModuleList(
            [
                SepEncStage(
                    in_channels=in_channels,
                    num_mha_heads=num_mha_heads,
                    global_dropout_rate=global_dropout_rate,
                    local_kernel_size=local_kernel_size,
                    local_dropout_rate=local_dropout_rate,
                    samp_kernel_size=samp_kernel_size,
                    down_conv=True,
                )
                for _ in range(num_stages)
            ]
        )
        self.bottleneck_g = SepEncStage(
            in_channels=in_channels,
            num_mha_heads=num_mha_heads,
            global_dropout_rate=global_dropout_rate,
            local_kernel_size=local_kernel_size,
            local_dropout_rate=local_dropout_rate,
            samp_kernel_size=samp_kernel_size,
            down_conv=False,
        )
        self.spk_split_block = SpkSplitStage(in_channels=in_channels, num_spk=num_spk)

        self.simple_fusion = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels * 2,
                    out_channels=in_channels,
                    kernel_size=1,
                )
                for _ in range(num_stages)
            ]
        )
        self.dec_stages = nn.ModuleList(
            [
                SepDecStage(
                    num_spk=num_spk,
                    in_channels=in_channels,
                    num_mha_heads=num_mha_heads,
                    global_dropout_rate=global_dropout_rate,
                    local_kernel_size=local_kernel_size,
                    local_dropout_rate=local_dropout_rate,
                    spk_attention_dropout_rate=spk_attention_dropout_rate,
                )
                for _ in range(num_stages)
            ]
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x, _ = self.pad_signal(input)
        len_x = x.shape[-1]
        down_len = len_x // 2**self.num_stages
        pos_seq = torch.arange(0, down_len, device=x.device, dtype=torch.long)
        pos_seq = pos_seq[:, None] - pos_seq[None, :]
        pos_k, _ = self.pos_emb(pos_seq)

        skip = []
        for idx in range(self.num_stages):
            x, skip_ = self.enc_stages[idx](x, pos_k)
            skip.append(self.spk_split_block(skip_))

        x, _ = self.bottleneck_g(x, pos_k)
        x = self.spk_split_block(x)

        each_stage_outputs = []
        for idx in range(self.num_stages):
            each_stage_outputs.append(x)
            idx_en = self.num_stages - (idx + 1)
            x = F.interpolate(x, size=skip[idx_en].shape[-1], mode="nearest")
            x = torch.cat([x, skip[idx_en]], dim=1)
            x = self.simple_fusion[idx](x)
            x, _ = self.dec_stages[idx](x, pos_k)

        return x, each_stage_outputs

    def pad_signal(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if input.dim() == 1:
            input = input.unsqueeze(0)
        elif input.dim() == 2:
            input = input.unsqueeze(1)
        elif input.dim() != 3:
            raise RuntimeError("Input can only be 1, 2, or 3 dimensional.")

        factor = 2**self.num_stages
        batch_size, ndim, nframe = input.size()
        padded_len = (nframe // factor + 1) * factor
        rest = 0 if nframe % factor == 0 else padded_len - nframe
        if rest > 0:
            pad = input.new_zeros(batch_size, ndim, rest)
            input = torch.cat([input, pad], dim=-1)
        return input, rest


class OutputLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_spk: int,
        masking: bool = False,
    ):
        super().__init__()
        self.masking = masking
        self.spe_block = Masking(in_channels, activation_mask="ReLU", concat_opt=None)
        self.num_spk = num_spk
        self.end_conv1x1 = nn.Sequential(
            nn.Linear(out_channels, 4 * out_channels),
            nn.GLU(),
            nn.Linear(2 * out_channels, in_channels),
        )

    def forward(self, x: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        x = x[..., : input.shape[-1]]
        x = x.permute(0, 2, 1)
        x = self.end_conv1x1(x)
        x = x.permute(0, 2, 1)
        batch_spk, channels, frames = x.shape
        batch = batch_spk // self.num_spk

        if self.masking:
            input = input.expand(self.num_spk, batch, channels, frames)
            input = input.transpose(0, 1).contiguous()
            input = input.view(batch * self.num_spk, channels, frames)
            x = self.spe_block(x, input)

        x = x.view(batch, self.num_spk, channels, frames)
        return x.transpose(0, 1)


class AudioDecoder(nn.ConvTranspose1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise RuntimeError("AudioDecoder only accepts 2D or 3D tensors.")
        return super().forward(x).squeeze(1)


class SepReformerSeparator(AbsSeparator):
    """SepReformer separator.

    This implementation follows the official SepReformer Base model structure and
    wraps it as an ESPnet enhancement separator. Use it with `encoder: same` and
    `decoder: same`, because the SepReformer audio encoder and decoder are
    internal to this separator.
    """

    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        num_stages: int = 4,
        audio_enc_in_channels: int = 1,
        audio_enc_out_channels: int = 256,
        audio_enc_kernel_size: int = 16,
        audio_enc_stride: int = 4,
        audio_enc_groups: int = 1,
        audio_enc_bias: bool = False,
        feature_dim: int = 128,
        feature_kernel_size: int = 1,
        feature_bias: bool = False,
        num_mha_heads: int = 8,
        maxlen: int = 2000,
        embed_v: bool = False,
        global_dropout_rate: float = 0.05,
        local_kernel_size: int = 65,
        local_dropout_rate: float = 0.05,
        samp_kernel_size: int = 5,
        spk_attention_dropout_rate: float = 0.05,
        output_aux: bool = True,
    ):
        super().__init__()
        del input_dim
        self._num_spk = num_spk
        self.num_stages = num_stages
        self.output_aux = output_aux

        self.audio_encoder = AudioEncoder(
            in_channels=audio_enc_in_channels,
            out_channels=audio_enc_out_channels,
            kernel_size=audio_enc_kernel_size,
            stride=audio_enc_stride,
            groups=audio_enc_groups,
            bias=audio_enc_bias,
        )
        self.feature_projector = FeatureProjector(
            num_channels=audio_enc_out_channels,
            in_channels=audio_enc_out_channels,
            out_channels=feature_dim,
            kernel_size=feature_kernel_size,
            bias=feature_bias,
        )
        self.separator = Separator(
            num_stages=num_stages,
            num_spk=num_spk,
            in_channels=feature_dim,
            num_mha_heads=num_mha_heads,
            maxlen=maxlen,
            embed_v=embed_v,
            global_dropout_rate=global_dropout_rate,
            local_kernel_size=local_kernel_size,
            local_dropout_rate=local_dropout_rate,
            samp_kernel_size=samp_kernel_size,
            spk_attention_dropout_rate=spk_attention_dropout_rate,
        )
        self.out_layer = OutputLayer(
            in_channels=audio_enc_out_channels,
            out_channels=feature_dim,
            num_spk=num_spk,
            masking=False,
        )
        self.audio_decoder = AudioDecoder(
            in_channels=audio_enc_out_channels,
            out_channels=1,
            kernel_size=audio_enc_kernel_size,
            stride=audio_enc_stride,
            bias=audio_enc_bias,
        )

        self.out_layer_bn = nn.ModuleList(
            [
                OutputLayer(
                    in_channels=audio_enc_out_channels,
                    out_channels=feature_dim,
                    num_spk=num_spk,
                    masking=True,
                )
                for _ in range(num_stages)
            ]
        )
        self.decoder_bn = nn.ModuleList(
            [
                AudioDecoder(
                    in_channels=audio_enc_out_channels,
                    out_channels=1,
                    kernel_size=audio_enc_kernel_size,
                    stride=audio_enc_stride,
                    bias=audio_enc_bias,
                )
                for _ in range(num_stages)
            ]
        )

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input: mixture waveform [B, T] or single-channel waveform [B, T, 1].
            ilens: input lengths [B].
            additional: reserved for ESPnet compatibility.

        Returns:
            During training with `output_aux=True`, a list of layer outputs is
            returned for `multilayer_pit`: [[spk1, spk2], ..., [spk1, spk2]].
            During evaluation, only the final [spk1, spk2] list is returned.
        """
        del additional
        if input.dim() == 3:
            assert input.shape[-1] == 1, "Only single-channel input is supported."
            input = input[..., 0]
        elif input.dim() != 2:
            raise RuntimeError("SepReformerSeparator expects [B, T] input.")

        target_length = input.shape[-1]
        encoder_output = self.audio_encoder(input)
        projected_feature = self.feature_projector(encoder_output)
        last_stage_output, each_stage_outputs = self.separator(projected_feature)

        out_layer_output = self.out_layer(last_stage_output, encoder_output)
        speech = [
            self._match_length(
                self.audio_decoder(out_layer_output[idx]),
                target_length,
            )
            for idx in range(self.num_spk)
        ]

        if self.training and self.output_aux:
            speech_aux = []
            for idx, each_stage_output in enumerate(each_stage_outputs):
                each_stage_output = F.interpolate(
                    each_stage_output,
                    size=encoder_output.shape[-1],
                    mode="nearest",
                )
                each_stage_output = self.out_layer_bn[idx](
                    each_stage_output,
                    encoder_output,
                )
                speech_aux.append(
                    [
                        self._match_length(
                            self.decoder_bn[idx](each_stage_output[spk]),
                            target_length,
                        )
                        for spk in range(self.num_spk)
                    ]
                )
            speech = speech_aux + [speech]

        return speech, ilens, OrderedDict()

    @staticmethod
    def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        if x.shape[-1] > target_length:
            return x[..., :target_length]
        if x.shape[-1] < target_length:
            return F.pad(x, (0, target_length - x.shape[-1]))
        return x

    @property
    def num_spk(self) -> int:
        return self._num_spk
