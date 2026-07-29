from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from espnet2.rir.rec_rir.model import BiSpatialNet


class BiSpatialNetPITCTFSplit(BiSpatialNet):
    """Two-source Rec-RIR with speaker-specific processing only in the CTF path."""

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
        num_spk = int(num_spk)
        if num_spk != 2:
            raise ValueError("BiSpatialNetPITCTFSplit currently supports num_spk=2")
        if dim_output_spch != num_spk * 2:
            raise ValueError(
                "dim_output_spch must be num_spk * 2 for real/imag output: "
                f"dim_output_spch={dim_output_spch}, num_spk={num_spk}"
            )
        if dim_output_CTF <= 0 or dim_output_CTF % 2 != 0:
            raise ValueError(
                "dim_output_CTF must be a positive even per-speaker output size: "
                f"{dim_output_CTF}"
            )

        super().__init__(
            dim_input=dim_input,
            dim_output_spch=dim_output_spch,
            dim_output_CTF=dim_output_CTF,
            dim_hidden=dim_hidden,
            dim_squeeze=dim_squeeze,
            num_freqs=num_freqs,
            num_layers_spch=num_layers_spch,
            num_layers_noise=num_layers_noise,
            num_layers_CTF=num_layers_CTF,
            encoder_kernel_size=encoder_kernel_size,
            dropout=dropout,
            kernel_size=kernel_size,
            conv_groups=conv_groups,
            norms=norms,
            padding=padding,
            full_share=full_share,
            attention=attention,
        )
        self.num_spk = num_spk
        self.ctf_taps = dim_output_CTF // 2
        self.speaker_split = nn.Linear(dim_hidden, self.num_spk * dim_hidden, bias=True)

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
        y_rev = self._decode_speech_like(self.decoder_rev(x_rev))

        for module in self.spch_layers:
            x = module(x)
        x_spch = x
        y_spch = self._decode_speech_like(self.decoder_spch(x_spch))

        x_spch = self._split_for_ctf(x_spch)
        x_rev = self._split_for_ctf(x_rev)
        batch, _, freq, frames, hidden = x_spch.shape
        x_spch = x_spch.reshape(batch * self.num_spk, freq, frames, hidden)
        x_rev = x_rev.reshape(batch * self.num_spk, freq, frames, hidden)

        x_ctf = self.compress_CTF(x_spch, x_rev)
        for module in self.ctf_layers:
            x_ctf = module(x_ctf)

        x_ctf = (x_ctf * self.weight_layer(x_ctf)).sum(-2).unsqueeze(2)
        x_ctf = x_ctf.reshape(batch, self.num_spk, freq, 1, hidden)
        if return_embedding:
            return x_ctf

        y_ctf = self.decoder_CTF(x_ctf.reshape(batch * self.num_spk, freq, 1, hidden))
        y_ctf = y_ctf.reshape(batch, self.num_spk, freq, 2, self.ctf_taps)
        y_ctf = y_ctf.permute(0, 1, 3, 2, 4).contiguous()
        return y_spch, y_ctf, y_rev

    def _split_for_ctf(self, x: Tensor) -> Tensor:
        batch, freq, frames, hidden = x.shape
        x = self.speaker_split(x)
        x = x.reshape(batch, freq, frames, self.num_spk, hidden)
        return x.permute(0, 3, 1, 2, 4).contiguous()

    def _decode_speech_like(self, x: Tensor) -> Tensor:
        batch, freq, frames, _ = x.shape
        x = x.reshape(batch, freq, frames, self.num_spk, 2)
        return x.permute(0, 3, 4, 1, 2).contiguous()
