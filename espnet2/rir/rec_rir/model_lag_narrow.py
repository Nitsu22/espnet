from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from espnet2.rir.rec_rir.model import BiSpatialNet


class LagAwareBiSpatialNet(BiSpatialNet):
    """Single-source Rec-RIR with an explicit CTF-lag sequence.

    The clean and reverberant embeddings are aligned at every CTF lag, fused
    with the original Rec-RIR scalar-weighted sum, and attention-pooled over
    valid time positions. The original narrow-band CTF layers then operate
    along the resulting lag axis.
    """

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
    ) -> None:
        if dim_output_CTF <= 0 or dim_output_CTF % 2 != 0:
            raise ValueError(
                "dim_output_CTF must be a positive even number because it "
                f"represents real/imaginary values for each lag: {dim_output_CTF}"
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
        self.num_ctf_lags = dim_output_CTF // 2
        self.decoder_CTF = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, 2),
        )

    def _pool_one_lag(
        self, clean_embedding: Tensor, reverb_embedding: Tensor
    ) -> Tensor:
        fused = self.compress_CTF(clean_embedding, reverb_embedding)
        return (fused * self.weight_layer(fused)).sum(dim=2)

    def _pool_lag_sequence(
        self, clean_embedding: Tensor, reverb_embedding: Tensor
    ) -> Tensor:
        if clean_embedding.shape != reverb_embedding.shape:
            raise ValueError(
                "Clean and reverberant embeddings must have the same shape: "
                f"{clean_embedding.shape} != {reverb_embedding.shape}"
            )

        num_frames = clean_embedding.shape[2]
        if num_frames < self.num_ctf_lags:
            raise ValueError(
                "The encoded input is too short for the requested CTF lag "
                f"sequence: frames={num_frames}, lags={self.num_ctf_lags}"
            )

        use_checkpoint = self.training and torch.is_grad_enabled()
        lag_embeddings = []
        for lag in range(self.num_ctf_lags):
            valid_frames = num_frames - lag
            clean_at_t = clean_embedding[:, :, :valid_frames, :]
            reverb_at_t_plus_lag = reverb_embedding[:, :, lag:, :]
            if use_checkpoint:
                pooled = checkpoint(
                    self._pool_one_lag,
                    clean_at_t,
                    reverb_at_t_plus_lag,
                    use_reentrant=False,
                )
            else:
                pooled = self._pool_one_lag(clean_at_t, reverb_at_t_plus_lag)
            lag_embeddings.append(pooled)
        return torch.stack(lag_embeddings, dim=2)

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

        x_CTF = self._pool_lag_sequence(x, x_rev)
        for module in self.ctf_layers:
            x_CTF = module(x_CTF)

        if return_embedding:
            return x_CTF
        y_CTF = self.decoder_CTF(x_CTF).permute(0, 3, 1, 2)
        return y_spch, y_CTF, y_rev
