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
        temporal_attn_type: str
            Attention type for temporal (frame) path. Must be either "mhsa" or "gated_fla".
            Default: "gated_fla" for FLA-TF-Locoformer.
        fla_p: int
            Power factor for focused function in Gated FLA (default: 3).
        fla_dwc_kernel: int
            Kernel size for depthwise convolution in Gated FLA (default: 7).
        fla_use_rope: bool
            Whether to use rotary positional encoding in Gated FLA (default: False).
    """

    def __init__(
        self,
        input_dim,
        num_spk: int = 2,
        n_layers: int = 6,
        # general setup
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
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
        # FLA related
        temporal_attn_type: str = "gated_fla",
        fla_p: int = 3,
        fla_dwc_kernel: int = 7,
        fla_use_rope: bool = False,
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

        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        if pos_enc == "nope":
            rope_freq = rope_time = None
        elif pos_enc == "rope":
            rope_freq = RotaryEmbedding(attention_dim // n_heads)
            rope_time = RotaryEmbedding(attention_dim // n_heads)
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
                    # FLA related
                    temporal_attn_type=temporal_attn_type,
                    fla_p=fla_p,
                    fla_dwc_kernel=fla_dwc_kernel,
                    fla_use_rope=fla_use_rope,
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
            additional (Dict or None): other data, currently unused in this model.

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

        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.conv(batch)  # [B, -1, T, F]

        # separation
        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        with torch.cuda.amp.autocast(enabled=False):
            batch = self.deconv(batch)  # [B, num_spk*2, T, F]
        batch = batch.view([n_batch, self.num_spk, 2, n_frames, n_freqs])

        batch = new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))
        batch = [batch[:, src] for src in range(self.num_spk)]

        return batch, ilens, OrderedDict()

    @property
    def num_spk(self):
        return self._num_spk


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope_freq,
        rope_time,
        # general setup
        emb_dim=128,
        norm_type="rmsgroupnorm",
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
        # FLA related
        temporal_attn_type: str = "gated_fla",
        fla_p: int = 3,
        fla_dwc_kernel: int = 7,
        fla_use_rope: bool = False,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

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
            # Frequency path uses MHSA
            attn_type="mhsa",
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
            # Temporal path uses Gated FLA
            attn_type=temporal_attn_type,
            fla_p=fla_p,
            fla_dwc_kernel=fla_dwc_kernel,
            fla_use_rope=fla_use_rope,
        )

    def forward(self, input):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        """

        if self.tf_order == "ft":
            output = self.freq_frame_process(input)
        else:
            output = self.frame_freq_process(input)

        return output

    def freq_frame_process(self, input):
        output = input.movedim(1, -1)  # (B, T, Q_old, H)
        output = self.freq_path(output)

        output = output.transpose(1, 2)  # (B, F, T, H)
        output = self.frame_path(output)
        return output.transpose(-1, 1)

    def frame_freq_process(self, input):
        # Input tensor, (n_batch, hidden, n_frame, n_freq)
        output = input.transpose(1, -1)  # (B, F, T, H)
        output = self.frame_path(output)

        output = output.transpose(1, 2)  # (B, T, F, H)
        output = self.freq_path(output)
        return output.movedim(-1, 1)


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        # general setup
        emb_dim=128,
        norm_type="rmsgroupnorm",
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
        # --- add ---
        attn_type: str = "mhsa",  # "mhsa" or "gated_fla"
        fla_p: int = 3,
        fla_dwc_kernel: int = 7,
        fla_use_rope: bool = False,  # decision #1: default False
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

        # Normalize ffn_type and ffn_hidden_dim to lists
        if isinstance(ffn_type, str):
            ffn_type = [ffn_type]
        if isinstance(ffn_hidden_dim, int):
            ffn_hidden_dim = [ffn_hidden_dim]

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
        # attn_norm is kept as in TF-Locoformer (shared hyper-params)
        if norm_type == "rmsgroupnorm":
            self.attn_norm = Norm[norm_type](num_groups, emb_dim, eps=eps)
        else:
            self.attn_norm = Norm[norm_type](emb_dim, eps=eps)

        if attn_type == "mhsa":
            self.attn = MultiHeadSelfAttention(
                emb_dim,
                attention_dim=attention_dim,
                n_heads=n_heads,
                rope=rope,
                dropout=dropout,
                flash_attention=flash_attention,
            )
        elif attn_type == "gated_fla":
            # Gated MLP uses RMSGroupNorm for TF-Locoformer variant (paper)
            # You requested: reuse TF-Locoformer hyper-params -> same num_groups, eps
            gate_norm = RMSGroupNorm(num_groups, emb_dim, eps=eps)

            self.attn = GatedFocusedLinearAttention(
                emb_dim=emb_dim,
                attention_dim=attention_dim,
                n_heads=n_heads,
                dropout=dropout,
                focused_factor_p=fla_p,  # paper p=3
                dwc_kernel_size=fla_dwc_kernel,  # paper k=7
                gate_norm=gate_norm,
                use_rope=fla_use_rope,  # default False
                rope=rope,
                eps=1e-6,
                qkv_bias=False,
                out_bias=False,
                gate_bias=False,
                dwc_bias=False,  # decision #3: default False
            )
        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}")

    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
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
        output = output.view([B * T, F, C])
        output = self.attn(output)
        output = output.view([B, T, F, C]) + input_

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

    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        if self.rope is not None:
            query, key = self.apply_rope(query, key)

        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
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


class GatedFocusedLinearAttention(nn.Module):
    """Gated Focused Linear Attention module for FLA-TF-Locoformer.

    This module implements the Gated Focused Linear Attention mechanism
    as described in the FLA-TF-Locoformer paper, replacing standard
    Multi-Head Self-Attention for temporal modeling.

    Args:
        emb_dim: int
            Embedding dimension.
        attention_dim: int
            Attention dimension (must be divisible by n_heads).
        n_heads: int
            Number of attention heads.
        dropout: float
            Dropout probability.
        focused_factor_p: int
            Power factor for focused function (default: 3).
        dwc_kernel_size: int
            Kernel size for depthwise convolution (default: 7).
        gate_norm: nn.Module
            Normalization module for gate (required, typically RMSGroupNorm).
        use_rope: bool
            Whether to use rotary positional encoding (default: False).
        rope: RotaryEmbedding or None
            Rotary embedding instance (ignored if use_rope=False).
        eps: float
            Small constant for numerical stability (default: 1e-6).
        qkv_bias: bool
            Whether to use bias in QKV projection (default: False).
        out_bias: bool
            Whether to use bias in output projection (default: False).
        gate_bias: bool
            Whether to use bias in gate linear layer (default: False).
        dwc_bias: bool
            Whether to use bias in depthwise convolution (default: False).
    """

    def __init__(
        self,
        emb_dim: int,
        attention_dim: int,
        n_heads: int,
        dropout: float,
        # paper hyper-params
        focused_factor_p: int = 3,
        dwc_kernel_size: int = 7,
        # gate norm for TF-Locoformer variant
        gate_norm: nn.Module = None,
        # implementation policy (your decisions #1 and #3)
        use_rope: bool = False,
        rope=None,
        eps: float = 1e-6,
        qkv_bias: bool = False,
        out_bias: bool = False,
        gate_bias: bool = False,
        dwc_bias: bool = False,
    ):
        super().__init__()
        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        assert gate_norm is not None

        self.n_heads = n_heads
        self.head_dim = attention_dim // n_heads
        self.attention_dim = attention_dim

        self.focused_factor_p = focused_factor_p
        self.eps = eps

        self.use_rope = use_rope
        self.rope = rope  # will be ignored if use_rope=False

        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=qkv_bias)

        # DWC1d(V) with same length
        padding = dwc_kernel_size // 2
        self.dwc = nn.Conv1d(
            in_channels=attention_dim,
            out_channels=attention_dim,
            kernel_size=dwc_kernel_size,
            padding=padding,
            groups=attention_dim,  # depthwise
            bias=dwc_bias,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(attention_dim, emb_dim, bias=out_bias),
            nn.Dropout(dropout),
        )

        # Gate: σ( Linear( Norm(X) ) )
        self.gate_norm = gate_norm
        self.gate_linear = nn.Linear(emb_dim, emb_dim, bias=gate_bias)

    def _focused_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Apply focused function φ_p: ReLU → pow(p) → norm scaling.

        Args:
            x: torch.Tensor
                Input tensor, (B, H, N, Dh)

        Returns:
            torch.Tensor: Focused kernel output, (B, H, N, Dh)
        """
        # x: (B, H, N, Dh)
        x = F.relu(x)
        x_p = x.pow(self.focused_factor_p)
        norm = x.norm(p=2, dim=-1, keepdim=True)
        norm_p = x_p.norm(p=2, dim=-1, keepdim=True)
        return (norm / (norm_p + self.eps)) * x_p

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Gated Focused Linear Attention.

        Args:
            x: torch.Tensor
                Input tensor, (B, N, emb_dim)

        Returns:
            torch.Tensor: Output tensor, (B, N, emb_dim)
        """
        # run in fp32 for stability (ReLU+pow+division)
        x = x.float()
        B, N, _ = x.shape

        qkv = self.qkv(x).view(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(0, 3, 1, 2, 4)  # (B, H, N, 3, Dh)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]

        if self.use_rope and (self.rope is not None):
            q = self.rope.rotate_queries_or_keys(q)
            k = self.rope.rotate_queries_or_keys(k)

        q = self._focused_kernel(q)
        k = self._focused_kernel(k)

        # Linear attention: (Q (K^T V)) / (Q (K^T 1))
        kv = torch.matmul(k.transpose(-1, -2), v)  # (B, H, Dh, Dh)
        num = torch.matmul(q, kv)  # (B, H, N, Dh)

        k_sum = k.sum(dim=2)  # (B, H, Dh)
        denom = torch.matmul(q, k_sum.unsqueeze(-1))  # (B, H, N, 1)
        out = num / (denom + self.eps)

        # + DWC1d(V)
        v_cat = v.permute(0, 2, 1, 3).reshape(B, N, self.attention_dim)  # (B, N, Dattn)
        v_dwc = self.dwc(v_cat.transpose(1, 2)).transpose(1, 2)  # (B, N, Dattn)
        v_dwc = v_dwc.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        out = out + v_dwc

        # concat heads -> proj
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.attention_dim)
        out = self.out_proj(out)

        # Gate
        gate = torch.sigmoid(self.gate_linear(self.gate_norm(x)))
        out = out * gate
        return out


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
