from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from espnet2.rir.rec_rir.model import FuseLayer, SpatialNetLayer, SpatialNetLayerNB


class BiSpatialNetPIT(nn.Module):
    """Rec-RIR BiSpatialNet variant with multiple source output heads."""

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
        num_spk: int = 2,
    ):
        super().__init__()
        self.num_spk = int(num_spk)
        if self.num_spk < 1:
            raise ValueError(f"num_spk must be positive: {num_spk}")
        if dim_output_spch != self.num_spk * 2:
            raise ValueError(
                "dim_output_spch must be num_spk * 2 for real/imag output: "
                f"dim_output_spch={dim_output_spch}, num_spk={self.num_spk}"
            )
        if dim_output_CTF % (self.num_spk * 2) != 0:
            raise ValueError(
                "dim_output_CTF must be divisible by num_spk * 2: "
                f"dim_output_CTF={dim_output_CTF}, num_spk={self.num_spk}"
            )
        self.ctf_taps = dim_output_CTF // (self.num_spk * 2)
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
        y_rev = self._decode_speech_like(self.decoder_rev(x))

        for module in self.spch_layers:
            x = module(x)
        y_spch = self._decode_speech_like(self.decoder_spch(x))

        x = self.compress_CTF(x, x_rev)
        for module in self.ctf_layers:
            x = module(x)

        x_CTF = (x * self.weight_layer(x)).sum(-2).unsqueeze(2)
        if return_embedding:
            return x_CTF
        batch, freq, _, _ = x_CTF.shape
        y_CTF = self.decoder_CTF(x_CTF).reshape(
            batch, freq, self.num_spk, 2, self.ctf_taps
        )
        y_CTF = y_CTF.permute(0, 2, 3, 1, 4).contiguous()
        return y_spch, y_CTF, y_rev

    def _decode_speech_like(self, x: Tensor) -> Tensor:
        batch, freq, frames, _ = x.shape
        x = x.reshape(batch, freq, frames, self.num_spk, 2)
        return x.permute(0, 3, 4, 1, 2).contiguous()
