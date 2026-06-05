from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter


class LinearGroup(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_groups: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x


class LayerNorm(nn.LayerNorm):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)
        output = super().forward(input)
        if self.seq_last:
            output = output.transpose(-1, 1)
        return output


class BatchNorm1d(nn.Module):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__()
        self.seq_last = seq_last
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if not self.seq_last:
            input = input.transpose(-1, -2)
        output = self.bn(input)
        if not self.seq_last:
            output = output.transpose(-1, -2)
        return output


class GroupNorm(nn.GroupNorm):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if not self.seq_last:
            input = input.transpose(-1, 1)
        output = super().forward(input)
        if not self.seq_last:
            output = output.transpose(-1, 1)
        return output


class GroupBatchNorm(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        group_size: Optional[int],
        share_along_sequence_dim: bool = False,
        seq_last: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
        dims_norm: Optional[List[int]] = None,
        dim_affine: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.seq_last = seq_last
        self.share_along_sequence_dim = share_along_sequence_dim
        self.dims_norm = dims_norm
        self.dim_affine = dim_affine

        if self.affine:
            if seq_last:
                weight = torch.empty([dim_hidden, 1])
                bias = torch.empty([dim_hidden, 1])
            else:
                weight = torch.empty([dim_hidden])
                bias = torch.empty([dim_hidden])
            if dim_affine is not None:
                if dim_affine >= 0:
                    raise ValueError(f"dim_affine must be negative: {dim_affine}")
                weight = weight.squeeze()
                bias = bias.squeeze()
                while dim_affine < -1:
                    weight = weight.unsqueeze(-1)
                    bias = bias.unsqueeze(-1)
                    dim_affine += 1
            self.weight = Parameter(weight)
            self.bias = Parameter(bias)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor, group_size: Optional[int] = None) -> Tensor:
        if self.group_size is not None:
            if group_size is not None and group_size != self.group_size:
                raise ValueError((group_size, self.group_size))
            group_size = self.group_size

        original_shape = x.shape
        if self.dims_norm is not None:
            var, mean = torch.var_mean(
                x, dim=self.dims_norm, unbiased=False, keepdim=True
            )
            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
            return output

        if group_size is None:
            raise ValueError("group_size must be given when dims_norm is None")

        if not self.seq_last:
            if x.ndim == 4:
                batch, group_size, seq, hidden = x.shape
            else:
                batch, seq, hidden = x.shape
                x = x.reshape(batch // group_size, group_size, seq, hidden)
            dims = (1, 2, 3) if self.share_along_sequence_dim else (1, 3)
        else:
            if x.ndim == 4:
                batch, group_size, hidden, seq = x.shape
            else:
                batch, hidden, seq = x.shape
                x = x.reshape(batch // group_size, group_size, hidden, seq)
            dims = (1, 2, 3) if self.share_along_sequence_dim else (1, 2)

        var, mean = torch.var_mean(x, dim=dims, unbiased=False, keepdim=True)
        output = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            output = output * self.weight + self.bias
        return output.reshape(original_shape)


class NoNorm(nn.Module):
    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


def new_norm(
    norm_type: str,
    dim_hidden: int,
    seq_last: bool,
    group_size: Optional[int] = None,
    num_groups: Optional[int] = None,
    dims_norm: Optional[List[int]] = None,
    dim_affine: Optional[int] = None,
) -> nn.Module:
    if norm_type.upper() == "LN":
        return LayerNorm(normalized_shape=dim_hidden, seq_last=seq_last)
    if norm_type.upper() == "NONE":
        return NoNorm(seq_last=seq_last)
    if norm_type.upper() == "GBN":
        return GroupBatchNorm(
            dim_hidden=dim_hidden,
            seq_last=seq_last,
            group_size=group_size,
            share_along_sequence_dim=False,
            dims_norm=dims_norm,
            dim_affine=dim_affine,
        )
    if norm_type == "GBNShare":
        return GroupBatchNorm(
            dim_hidden=dim_hidden,
            seq_last=seq_last,
            group_size=group_size,
            share_along_sequence_dim=True,
            dims_norm=dims_norm,
            dim_affine=dim_affine,
        )
    if norm_type.upper() == "BN":
        return BatchNorm1d(num_features=dim_hidden, seq_last=seq_last)
    if norm_type.upper() == "GN":
        return GroupNorm(
            num_groups=num_groups,
            num_channels=dim_hidden,
            seq_last=seq_last,
        )
    raise ValueError(f"Unsupported normalization: {norm_type}")


def _build_mamba(dim_hidden: int, attention: str) -> nn.Module:
    try:
        from mamba_ssm import Mamba
    except Exception as e:
        raise ImportError(
            "Rec-RIR requires mamba_ssm. Install the official Rec-RIR "
            "dependencies in the runtime environment before training."
        ) from e

    if not attention.startswith("mamba(") or not attention.endswith(")"):
        raise ValueError(f"Unsupported Rec-RIR attention: {attention}")
    d_state, conv_kernel = [int(v) for v in attention[6:-1].split(",")]
    return Mamba(
        d_model=dim_hidden,
        d_state=d_state,
        d_conv=conv_kernel,
        expand=2,
    )


class SpatialNetLayer(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "LN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full: Optional[nn.Module] = None,
        attention: str = "mamba(16,4)",
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]

        self.fconv1 = nn.ModuleList(
            [
                new_norm(
                    norms[3],
                    dim_hidden,
                    seq_last=True,
                    group_size=None,
                    num_groups=f_conv_groups,
                ),
                nn.Conv1d(
                    dim_hidden,
                    dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
            ]
        )
        self.norm_full = new_norm(
            norms[5],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=f_conv_groups,
        )
        self.full_share = full is not None
        self.squeeze = nn.Sequential(
            nn.Conv1d(dim_hidden, dim_squeeze, kernel_size=1),
            nn.SiLU(),
        )
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = (
            LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze)
            if full is None
            else full
        )
        self.unsqueeze = nn.Sequential(
            nn.Conv1d(dim_squeeze, dim_hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.fconv2 = nn.ModuleList(
            [
                new_norm(
                    norms[4],
                    dim_hidden,
                    seq_last=True,
                    group_size=None,
                    num_groups=f_conv_groups,
                ),
                nn.Conv1d(
                    dim_hidden,
                    dim_hidden,
                    kernel_size=f_kernel_size,
                    groups=f_conv_groups,
                    padding="same",
                    padding_mode=padding,
                ),
                nn.PReLU(dim_hidden),
            ]
        )

        self.norm_mhsa = new_norm(
            norms[0],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.mhsa_f = _build_mamba(dim_hidden, attention)
        self.dropout_mhsa = nn.Dropout(dropout[0])

        self.norm_tconvffn = new_norm(
            norms[1],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.tconvffn_b = _build_mamba(dim_hidden, attention)
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        x = x + self._mamba(x, self.mhsa_f, self.norm_mhsa, self.dropout_mhsa)
        x = x + self._mamba(
            x.flip(-2), self.tconvffn_b, self.norm_tconvffn, self.dropout_tconvffn
        ).flip(-2)
        return x

    def _mamba(
        self, x: Tensor, mamba: nn.Module, norm: nn.Module, dropout: nn.Module
    ) -> Tensor:
        batch, freq, frames, hidden = x.shape
        x = norm(x).reshape(batch * freq, frames, hidden)
        x = mamba(x)
        return dropout(x.reshape(batch, freq, frames, hidden))

    def _fconv(self, modules: nn.ModuleList, x: Tensor) -> Tensor:
        batch, freq, frames, hidden = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch * frames, hidden, freq).contiguous()
        for module in modules:
            if isinstance(module, GroupBatchNorm):
                x = module(x, group_size=frames)
            else:
                x = module(x)
        return x.reshape(batch, frames, hidden, freq).permute(0, 3, 1, 2).contiguous()

    def _full(self, x: Tensor) -> Tensor:
        batch, freq, frames, hidden = x.shape
        x = self.norm_full(x).permute(0, 2, 3, 1).reshape(batch * frames, hidden, freq)
        x = self.squeeze(x.contiguous())
        if self.dropout_full is not None:
            x = x.view(batch, frames, -1, freq)
            x = self.dropout_full(x.transpose(1, 3)).transpose(1, 3)
            x = x.reshape(batch * frames, -1, freq).contiguous()
        x = self.unsqueeze(self.full(x))
        return x.view(batch, frames, hidden, freq).permute(0, 3, 1, 2).contiguous()


class SpatialNetLayerNB(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "LN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full: Optional[nn.Module] = None,
        attention: str = "mamba(16,4)",
    ) -> None:
        super().__init__()
        t_conv_groups = conv_groups[1]
        self.full_share = full is not None
        self.norm_mamba_t_f = new_norm(
            norms[0],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.mamba_t_f = _build_mamba(dim_hidden, attention)
        self.dropout_mamba_t_f = nn.Dropout(dropout[0])
        self.norm_mamba_t_b = new_norm(
            norms[1],
            dim_hidden,
            seq_last=False,
            group_size=None,
            num_groups=t_conv_groups,
        )
        self.mamba_t_b = _build_mamba(dim_hidden, attention)
        self.dropout_mamba_t_b = nn.Dropout(dropout[1])

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._mamba(
            x, self.mamba_t_f, self.norm_mamba_t_f, self.dropout_mamba_t_f
        )
        x = x + self._mamba(
            x.flip(-2),
            self.mamba_t_b,
            self.norm_mamba_t_b,
            self.dropout_mamba_t_b,
        ).flip(-2)
        return x

    def _mamba(
        self, x: Tensor, mamba: nn.Module, norm: nn.Module, dropout: nn.Module
    ) -> Tensor:
        batch, freq, frames, hidden = x.shape
        x = norm(x).reshape(batch * freq, frames, hidden)
        x = mamba(x)
        return dropout(x.reshape(batch, freq, frames, hidden))


class FuseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.alpha * x + self.beta * y


class BiSpatialNet(nn.Module):
    """Official Rec-RIR BiSpatialNet adapted for ESPnet."""

    def __init__(
        self,
        dim_input: int,
        dim_output_spch: int,
        dim_output_CTF: int,
        dim_hidden: int,
        dim_squeeze: int,
        num_freqs: int,
        num_layers_spch: int,
        num_layers_noise: int,
        num_layers_CTF: int,
        encoder_kernel_size: int = 1,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
        padding: str = "zeros",
        full_share: int = 0,
        attention: str = "mamba(16,4)",
    ):
        super().__init__()
        self.padding_size = (0, (encoder_kernel_size - 1) // 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=dim_input,
                out_channels=dim_hidden,
                padding=0,
                kernel_size=(1, encoder_kernel_size),
            ),
            nn.PReLU(),
        )

        full = None
        spch_layers = []
        for layer_idx in range(num_layers_spch):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if layer_idx > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            spch_layers.append(layer)
        self.spch_layers = nn.ModuleList(spch_layers)

        full = None
        noise_layers = []
        for layer_idx in range(num_layers_noise):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if layer_idx > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            noise_layers.append(layer)
        self.noise_layers = nn.ModuleList(noise_layers)

        full = None
        ctf_layers = []
        for layer_idx in range(num_layers_CTF):
            layer = SpatialNetLayerNB(
                dim_hidden=dim_hidden,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if layer_idx > full_share else None,
                attention=attention,
            )
            if hasattr(layer, "full"):
                full = layer.full
            ctf_layers.append(layer)
        self.ctf_layers = nn.ModuleList(ctf_layers)

        self.decoder_spch = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_output_spch),
        )
        self.decoder_rev = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_output_spch),
        )
        self.decoder_CTF = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_output_CTF),
        )
        self.compress_CTF = FuseLayer()
        self.weight_layer = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Softmax(dim=2),
        )

    def forward(self, input: Tensor, return_embedding: bool = False):
        input_pad = torch.nn.functional.pad(
            input,
            (
                self.padding_size[1],
                self.padding_size[1],
                self.padding_size[0],
                self.padding_size[0],
            ),
            mode="constant",
            value=0,
        )
        x = self.encoder(input_pad).permute(0, 2, 3, 1)

        for module in self.noise_layers:
            x = module(x)
        x_rev = x
        y_rev = self.decoder_rev(x).permute(0, 3, 1, 2)

        for module in self.spch_layers:
            x = module(x)
        y_spch = self.decoder_spch(x).permute(0, 3, 1, 2)

        x = self.compress_CTF(x, x_rev)
        for module in self.ctf_layers:
            x = module(x)

        x_CTF = (x * self.weight_layer(x)).sum(-2).unsqueeze(2)
        if return_embedding:
            return x_CTF
        batch, freq, _, _ = x_CTF.shape
        y_CTF = self.decoder_CTF(x_CTF).reshape([batch, freq, 2, -1])
        y_CTF = y_CTF.permute(0, 2, 1, 3)
        return y_spch, y_CTF, y_rev

