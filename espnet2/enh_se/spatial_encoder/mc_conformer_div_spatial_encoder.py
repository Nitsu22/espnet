"""
MC-Conformer Spatial Encoder (div version with separate SC/MC branches).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from espnet2.enh_se.spatial_encoder.abs_spatial_encoder import AbsSpatialEncoder
from espnet2.enh_se.spatial_encoder.mc_conformer import MCConformerSpatialEncoder


class MCConformerDivSpatialEncoder(AbsSpatialEncoder):
    """MC-Conformer Spatial Encoder with separate parameters for SC and MC."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_channels_mc: int = 2,
        freq_bins: int = 256,
        attention_heads: int = 4,
        num_blocks: int = 3,
        ffn_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout_rate: float = 0.1,
        eps: float = 1e-8,
        l2_norm_eps: float = 1e-8,
    ):
        super().__init__()
        self.num_channels_mc = num_channels_mc
        self.l2_norm_eps = l2_norm_eps

        # Separate SC and MC encoders
        self.sc_encoder = MCConformerSpatialEncoder(
            embed_dim=embed_dim,
            num_channels_mc=1,
            freq_bins=freq_bins,
            attention_heads=attention_heads,
            num_blocks=num_blocks,
            ffn_expansion=ffn_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
            eps=eps,
        )
        self.mc_encoder = MCConformerSpatialEncoder(
            embed_dim=embed_dim,
            num_channels_mc=num_channels_mc,
            freq_bins=freq_bins,
            attention_heads=attention_heads,
            num_blocks=num_blocks,
            ffn_expansion=ffn_expansion,
            conv_kernel_size=conv_kernel_size,
            dropout_rate=dropout_rate,
            eps=eps,
        )

    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=self.l2_norm_eps)

    def _forward_sc(
        self, input: torch.Tensor, ilens: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if input.ndim == 3:
            # [B, T, F] -> [B, T, 1, F]
            input = input.unsqueeze(2)
        elif input.ndim == 4:
            if input.shape[2] != 1:
                raise ValueError(
                    f"Expected 1 channel for SC, but got {input.shape[2]}"
                )
        else:
            raise ValueError(f"Unexpected input dim for SC: {input.ndim}")

        emb = self.sc_encoder(input, ilens, num_channels=1, pooling=True)
        return self._l2_normalize(emb)

    def _forward_mc(
        self, input: torch.Tensor, ilens: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if input.ndim != 4:
            raise ValueError(f"Expected 4D input for MC, but got {input.ndim}D")
        if input.shape[2] != self.num_channels_mc:
            raise ValueError(
                f"Expected {self.num_channels_mc} channels, but got {input.shape[2]}"
            )
        emb = self.mc_encoder(
            input, ilens, num_channels=self.num_channels_mc, pooling=True
        )
        return self._l2_normalize(emb)

    def forward(
        self,
        input: torch.Tensor,
        ilens: Optional[torch.Tensor],
        num_channels: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input: [B, T, F] or [B, T, C, F] - STFT spectrum (complex)
            ilens: [B] - input lengths
            num_channels: Number of input channels (1 for SC, num_channels_mc for MC).
        Returns:
            embedding: [B, embed_dim] - L2-normalized spatial embedding
        """
        if num_channels is not None:
            if num_channels == 1:
                return self._forward_sc(input, ilens)
            if num_channels == self.num_channels_mc:
                return self._forward_mc(input, ilens)
            raise ValueError(
                f"Unexpected num_channels: {num_channels}. "
                f"Expected 1 or {self.num_channels_mc}."
            )

        if input.ndim == 3:
            return self._forward_sc(input, ilens)
        if input.ndim == 4:
            if input.shape[2] == 1:
                return self._forward_sc(input, ilens)
            if input.shape[2] == self.num_channels_mc:
                return self._forward_mc(input, ilens)
            raise ValueError(
                f"Unexpected number of channels: {input.shape[2]}. "
                f"Expected 1 or {self.num_channels_mc}."
            )
        raise ValueError(f"Unexpected input dim: {input.ndim}. Expected 3 or 4.")


__all__ = ["MCConformerDivSpatialEncoder"]
