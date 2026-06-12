# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0


import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet2.enh.layers.complex_utils import new_complex_like
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding

from espnet2.enh.separator.abs_separator import AbsSeparator

is_torch_2_0_plus = V(torch.__version__) >= V("2.0.0")


class RIRTemporalBiasEncoder(nn.Module):
    """Encode RI RIR features into layer/head-wise temporal lag bias."""

    def __init__(
        self,
        in_channels: int,
        conv_channels: int,
        n_layers: int,
        n_heads: int,
        gru_hidden_size: int = 96,
        gru_num_layers: int = 2,
        dropout: float = 0.0,
        eps: float = 1.0e-5,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, conv_channels, eps=eps),
            nn.PReLU(),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(gru_hidden_size * 2, n_layers * n_heads)

    def forward(self, rir: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            rir: RI RIR feature, [B, 2 * num_spk, T2, F]

        Returns:
            Layer/head-wise lag bias values, [B, n_layers, n_heads, T2, F]
        """
        bsz, _, n_lags, n_freqs = rir.shape
        output = self.conv(rir)  # [B, C, T2, F]
        output = output.permute(0, 3, 2, 1).contiguous()
        output = output.view(bsz * n_freqs, n_lags, output.shape[-1])
        output, _ = self.gru(output)
        output = self.proj(output)
        output = output.view(bsz, n_freqs, n_lags, self.n_layers, self.n_heads)
        output = output.permute(0, 3, 4, 2, 1)
        return torch.tanh(output)


class TFLocoformerSeparator(AbsSeparator):
    """TF-Locoformer model presented in [1].

    Reference:
    [1] Kohei Saijo, Gordon Wichern, François G. Germain, Zexu Pan, and Jonathan Le Roux,
    "TF-Locoformer: Transformer with Local Modeling by Convolution for Speech Separation
    and Enhancement," in Proc. International Workshop on Acoustic Signal Enhancement (IWAENC),
    Sep. 2024.

    Args:
        input_dim: int
            placeholder, not used
        num_spk: int
            number of output sources/speakers.
        n_layers: int
            number of Locoformer blocks.
        emb_dim: int
            Size of hidden dimension in the encoding Conv2D.
        norm_type: str
            Normalization layer. Must be either "layernorm" or "rmsgroupnorm".
        num_groups: int
            Number of groups in RMSGroupNorm layer.
        tf_order: str
            Order of frequency and temporal modeling. Must be either "ft" or "tf".
        n_heads: int
            Number of heads in multi-head self-attention.
        flash_attention: bool
            Whether to use flash attention. Only compatible with half precision.
        ffn_type: str or list
            Feed-forward network (FFN)-type chosen from "conv1d" or "swiglu_conv1d".
            Giving the list (e.g., ["conv1d", "conv1d"]) makes the model Macaron-style.
        ffn_hidden_dim: int or list
            Number of hidden dimensions in FFN.
            Giving the list (e.g., [256, 256]) makes the model Macaron-style.
        conv1d_kernel: int
            Kernel size in Conv1d.
        conv1d_shift: int
            Shift size of Conv1d kernel.
        dropout: float
            Dropout probability.
        eps: float
            Small constant for normalization layer.
    """

    def __init__(
        self,
        input_dim,
        num_spk: int = 2,
        n_layers: int = 6,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgrouporm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        pos_enc: str = "rope",
        # ffn related
        ffn_type: Union[str, list] = "swiglu_conv1d",
        ffn_hidden_dim: Union[int, list] = 384,
        conv1d_kernel: int = 4,
        conv1d_shift: int = 1,
        dropout: float = 0.0,
        # others
        eps: float = 1.0e-5,
    ):
        super().__init__()
        assert is_torch_2_0_plus, "Support only pytorch >= 2.0.0"

        self._num_spk = num_spk
        self.n_layers = n_layers

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),  # gLN
        )
        self.rir_bias_encoder = RIRTemporalBiasEncoder(
            in_channels=num_spk * 2,
            conv_channels=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            eps=eps,
        )

        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        if pos_enc == "nope":
            rope_freq = rope_time = None
        elif pos_enc == "rope":
            # Disable RoPE internal caching so that length-dependent tensors are not
            # materialized as DDP-synced buffers.
            rope_freq = RotaryEmbedding(attention_dim // n_heads, cache_if_possible=False)
            rope_time = RotaryEmbedding(attention_dim // n_heads, cache_if_possible=False)
        else:
            raise ValueError(f"Unsupported positional encoding: {pos_enc}")

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                TFLocoformerBlock(
                    rope_freq,
                    rope_time,
                    # general setup
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    tf_order=tf_order,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=eps,
                )
            )

        self.deconv = nn.ConvTranspose2d(emb_dim, num_spk * 2, ks, padding=padding)

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched single-channel audio tensor with
                in TF-domain [B, T, F]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): must include "rir_feat" [B, T2, num_spk, F].

        Returns:
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len num_spk
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in the output.
        """
        if input.ndim == 3:
            # in case the input does not have channel dimension
            batch0 = input.unsqueeze(1)
        elif input.ndim == 4:
            assert input.shape[1] == 1, "Only monaural input is supported."
            batch0 = input.transpose(1, 2)  # [B, M, T, F]
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape
        rir_feat = self._get_rir_feat(additional, n_batch, n_freqs)

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch)  # [B, -1, T, F]
            rir_bias = self.rir_bias_encoder(rir_feat)

        # separation
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch, rir_bias[:, ii])  # [B, -1, T, F]

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.deconv(batch)  # [B, num_spk*2, T, F]
        batch = batch.view([n_batch, self.num_spk, 2, n_frames, n_freqs])

        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))
        batch = [batch[:, src] for src in range(self.num_spk)]

        return batch, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self._num_spk

    def _get_rir_feat(
        self,
        additional: Optional[Dict],
        n_batch: int,
        n_freqs: int,
    ) -> torch.Tensor:
        if additional is None or "rir_feat" not in additional:
            raise ValueError("rir_feat is required in additional for RIR temporal bias.")

        rir = additional["rir_feat"]
        if rir.ndim != 4:
            raise ValueError(
                "rir_feat must be complex STFT [B, T2, num_spk, F], "
                f"but got {tuple(rir.shape)}"
            )
        if rir.shape[0] != n_batch or rir.shape[2] != self.num_spk:
            raise ValueError(
                "rir_feat must have shape [B, T2, num_spk, F]: "
                f"got {tuple(rir.shape)}, B={n_batch}, num_spk={self.num_spk}"
            )
        if rir.shape[3] != n_freqs:
            raise ValueError(
                f"RIR frequency bins must match speech: {rir.shape[3]} != {n_freqs}"
            )
        if not torch.is_complex(rir):
            raise ValueError("rir_feat must be a complex tensor.")

        rir = rir.transpose(1, 2)  # [B, num_spk, T2, F]
        return torch.cat((rir.real, rir.imag), dim=1)  # [B, 2*num_spk, T2, F]


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope_freq,
        rope_time,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        tf_order="ft",
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift
        self.rir_bias_scale = 0.1

        self.freq_path = LocoformerBlock(
            rope_freq,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            rope_time,
            # general setup
            emb_dim=emb_dim,
            norm_type=norm_type,
            num_groups=num_groups,
            # self-attention related
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            # ffn related
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            conv1d_kernel=conv1d_kernel,
            conv1d_shift=conv1d_shift,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, input, rir_bias_layer=None):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        rir_bias_layer: torch.Tensor or None
            RIR temporal lag bias for this layer, [B, H, T2, F]
        """

        if self.tf_order == "ft":
            output = self.freq_frame_process(input, rir_bias_layer)
        else:
            output = self.frame_freq_process(input, rir_bias_layer)

        return output

    def freq_frame_process(self, input, rir_bias_layer=None):
        output = input.movedim(1, -1)  # (B, T, Q_old, H)
        output = self.freq_path(output)

        output = output.transpose(1, 2)  # (B, F, T, H)
        attn_bias = self._create_temporal_attn_bias(rir_bias_layer, output.shape[2])
        output = self.frame_path(output, attn_bias=attn_bias)
        return output.transpose(-1, 1)

    def frame_freq_process(self, input, rir_bias_layer=None):
        # Input tensor, (n_batch, hidden, n_frame, n_freq)
        output = input.transpose(1, -1)  # (B, F, T, H)
        attn_bias = self._create_temporal_attn_bias(rir_bias_layer, output.shape[2])
        output = self.frame_path(output, attn_bias=attn_bias)

        output = output.transpose(1, 2)  # (B, T, F, H)
        output = self.freq_path(output)
        return output.movedim(-1, 1)

    def _create_temporal_attn_bias(self, rir_bias_layer, n_frames: int):
        if rir_bias_layer is None:
            return None

        if rir_bias_layer.ndim != 4:
            raise ValueError(
                "rir_bias_layer must have shape [B, H, T2, F], "
                f"but got {tuple(rir_bias_layer.shape)}"
            )

        bsz, n_heads, n_lags, n_freqs = rir_bias_layer.shape
        max_lag = min(n_lags, n_frames)
        if max_lag <= 1:
            return rir_bias_layer.new_zeros(
                bsz,
                n_freqs,
                n_heads,
                n_frames,
                n_frames,
            )

        frame_idx = torch.arange(n_frames, device=rir_bias_layer.device)
        lag = (frame_idx[:, None] - frame_idx[None, :]).abs()
        valid = (lag > 0) & (lag < max_lag)
        safe_lag = lag.clamp(max=max_lag - 1)

        bias = rir_bias_layer.permute(0, 3, 1, 2)[..., safe_lag]
        bias = bias.masked_fill(~valid, 0.0)
        return bias * self.rir_bias_scale


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        # general setup
        emb_dim=128,
        norm_type="rmsgrouporm",
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {
            "conv1d": ConvDeconv1d,
            "swiglu_conv1d": SwiGLUConvDeconv1d,
        }
        Norm = {
            "layernorm": nn.LayerNorm,
            "rmsgroupnorm": RMSGroupNorm,
        }
        assert norm_type in Norm, norm_type

        self.macaron_style = isinstance(ffn_type, list) and len(ffn_type) == 2
        if self.macaron_style:
            assert (
                isinstance(ffn_hidden_dim, list) and len(ffn_hidden_dim) == 2
            ), "Two FFNs required when using Macaron-style model"

        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            assert f_type in FFN, f_type
            if norm_type == "rmsgroupnorm":
                self.ffn_norm.append(Norm[norm_type](num_groups, emb_dim, eps=eps))
            else:
                self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )

        # initialize self-attention
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x, attn_bias=None):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
            attn_bias: torch.Tensor or None
                Additive self-attention bias, [B, seq1, n_heads, seq2, seq2]
        """
        B, T, F, C = x.shape

        if self.macaron_style:
            # FFN before self-attention
            # Note that this implementation does not include the 1/2 factor described in the paper.
            # Experiments in the paper did use the 1/2 factor, but we removed it by mistake in this
            # implementation. We found that the 1/2 factor does not impact final performance, and
            # thus decided to keep the current implementation for consistency with the pre-trained
            # models that we provide.
            input_ = x
            output = self.ffn_norm[-1](x)  # [B, T, F, C]
            output = self.ffn[-1](output)  # [B, T, F, C]
            output = output + input_
        else:
            output = x

        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = output.reshape([B * T, F, C])
        if attn_bias is not None:
            expected_shape = (B, T, self.attn.n_heads, F, F)
            if tuple(attn_bias.shape) != expected_shape:
                raise ValueError(
                    "attn_bias must have shape [B, seq1, n_heads, seq2, seq2]: "
                    f"expected {expected_shape}, got {tuple(attn_bias.shape)}"
                )
            attn_bias = attn_bias.contiguous().view(B * T, self.attn.n_heads, F, F)
        output = self.attn(output, attn_bias=attn_bias)
        output = output.reshape([B, T, F, C]) + input_

        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output + input_

        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input, attn_bias=None):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        if self.rope is not None:
            query, key = self.apply_rope(query, key)

        if attn_bias is not None:
            if tuple(attn_bias.shape) != query.shape[:-1] + (key.shape[-2],):
                raise ValueError(
                    "attn_bias must have shape [B, H, query_len, key_len]: "
                    f"expected {tuple(query.shape[:-1] + (key.shape[-2],))}, "
                    f"got {tuple(attn_bias.shape)}"
                )
            attn_bias = attn_bias.to(device=query.device, dtype=query.dtype)

        flash_attention_config = self.flash_attention_config
        if attn_bias is not None and self.flash_attention_config["enable_flash"]:
            flash_attention_config = dict(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=True,
            )

        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with torch.backends.cuda.sdp_kernel(**flash_attention_config):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_bias,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class ConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.diff_ks = conv1d_kernel - conv1d_shift

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """ConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output
