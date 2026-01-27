"""
MC-Conformer Spatial Encoder implementation.
"""

from typing import Optional

import torch
import torch.nn as nn

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh_se.spatial_encoder.abs_spatial_encoder import AbsSpatialEncoder
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from torch_complex.tensor import ComplexTensor


class RIStack(nn.Module):
    """Convert complex STFT to real-valued RI stack.

    Args:
        x: [B, C, F, T] (complex)
    Returns:
        x_ri: [B, 2*C, F, T] (real)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x.real
        x_imag = x.imag
        return torch.cat([x_real, x_imag], dim=1)


class LocalProcess(nn.Module):
    """Local process CNN block (Fig. 3(c))."""

    def __init__(
        self,
        freq_bins: int = 256,
        in_channels: int = 4,
        mid_channels: int = 64,
        bottleneck_channels: int = 4,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.freq_bins = freq_bins

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv4 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn4 = nn.BatchNorm2d(bottleneck_channels)
        self.conv5 = nn.Conv2d(
            in_channels=bottleneck_channels,
            out_channels=embed_dim,
            kernel_size=(freq_bins, 1),
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2*C, F, T] (real)
        Returns:
            y: [B, T, embed_dim]
        """
        if x.shape[2] != self.freq_bins:
            raise ValueError(
                f"Expected freq_bins={self.freq_bins}, but got F={x.shape[2]}"
            )

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)  # [B, embed_dim, 1, T]

        x = x.squeeze(2)  # [B, embed_dim, T]
        x = x.permute(0, 2, 1)  # [B, T, embed_dim]
        return x


class MCConformerSpatialEncoder(AbsSpatialEncoder):
    """MC-Conformer Spatial Encoder."""

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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels_mc = num_channels_mc
        self.freq_bins = freq_bins
        self.eps = eps

        self.ri_stack = RIStack()
        self.local = LocalProcess(
            freq_bins=freq_bins,
            in_channels=2 * num_channels_mc,
            mid_channels=64,
            bottleneck_channels=2 * num_channels_mc,
            embed_dim=embed_dim,
        )

        self.conformer = ConformerEncoder(
            idim=embed_dim,
            attention_dim=embed_dim,
            attention_heads=attention_heads,
            linear_units=embed_dim * ffn_expansion,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=dropout_rate,
            input_layer=None,
            normalize_before=True,
            concat_after=False,
            positionwise_layer_type="linear",
            positionwise_conv_kernel_size=1,
            macaron_style=True,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=conv_kernel_size,
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by mean magnitude of the first microphone channel.

        Args:
            x: [B, T, C, F] (complex)
        Returns:
            x_norm: [B, T, C, F] (complex)
        """
        mic1 = x[:, :, 0, :]  # [B, T, F]
        mag = torch.sqrt(mic1.real ** 2 + mic1.imag ** 2 + self.eps)
        denom = mag.mean(dim=(1, 2), keepdim=True) + self.eps  # [B, 1, 1]
        denom = denom.unsqueeze(2)  # [B, 1, 1, 1]
        if isinstance(x, ComplexTensor):
            return ComplexTensor(x.real / denom, x.imag / denom)
        return x / denom

    def _mean_pool(
        self, x: torch.Tensor, ilens: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean pool over time axis with optional length masking."""
        if ilens is None:
            return x.mean(dim=1)
        if ilens.ndim != 1:
            raise ValueError(f"ilens must be 1D, but got shape {ilens.shape}")
        if ilens.shape[0] != x.shape[0]:
            raise ValueError(
                f"ilens length {ilens.shape[0]} does not match batch size {x.shape[0]}"
            )

        max_len = x.shape[1]
        ilens = ilens.to(x.device).long()
        if torch.any(ilens > max_len):
            raise ValueError(
                f"ilens has values larger than max_len={max_len}: {ilens}"
            )

        seq_range = torch.arange(max_len, device=x.device)
        mask = (seq_range.unsqueeze(0) < ilens.unsqueeze(1)).to(x.dtype)  # [B, T]
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        x_sum = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return x_sum / denom

    def forward(
        self,
        input: torch.Tensor,
        ilens: Optional[torch.Tensor],
        num_channels: Optional[int] = None,
        pooling: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input: [B, T, C, F] - STFT spectrum (complex)
            ilens: [B] - input lengths
            num_channels: Expected number of channels. If None, uses self.num_channels_mc from config.
            pooling: If True, return mean-pooled embedding [B, 256]
            **kwargs: Additional keyword arguments (for compatibility)
        Returns:
            embedding: [B, T, 256] by default, or [B, 256] if pooling=True
        """
        if not is_complex(input):
            raise TypeError("input must be a complex tensor.")
        if input.ndim != 4:
            raise ValueError(
                f"Expected 4D input [B, T, C, F], but got shape {input.shape}"
            )

        # num_channelsが指定されていない場合は、configから取得
        if num_channels is None:
            num_channels = self.num_channels_mc

        bsz, n_frames, n_ch, n_freq = input.shape
        if num_channels != self.num_channels_mc:
            raise ValueError(
                f"num_channels must be {self.num_channels_mc}, but got {num_channels}"
            )
        if n_ch != self.num_channels_mc:
            raise ValueError(
                f"Expected {self.num_channels_mc} channels, but got {n_ch}"
            )

        # Remove Nyquist bin (F=257 -> F=256)
        if n_freq == self.freq_bins + 1:
            input = input[..., : self.freq_bins]
            n_freq = self.freq_bins
        if n_freq != self.freq_bins:
            raise ValueError(
                f"Expected F={self.freq_bins} after Nyquist removal, but got F={n_freq}"
            )

        x = self._normalize(input)

        # [B, T, C, F] -> [B, C, F, T]
        x = x.permute(0, 2, 3, 1)
        x = self.ri_stack(x)  # [B, 2C, F, T]

        x = self.local(x)  # [B, T, embed_dim]

        if ilens is None:
            pad_mask = torch.ones(
                (bsz, 1, n_frames), dtype=torch.bool, device=x.device
            )
        else:
            if ilens.ndim != 1:
                raise ValueError(f"ilens must be 1D, but got shape {ilens.shape}")
            if ilens.shape[0] != bsz:
                raise ValueError(
                    f"ilens length {ilens.shape[0]} does not match batch size {bsz}"
                )
            if torch.any(ilens > n_frames):
                raise ValueError(
                    f"ilens has values larger than n_frames={n_frames}: {ilens}"
                )
            pad_mask = make_non_pad_mask(ilens).unsqueeze(1).to(x.device)

        x, _ = self.conformer(x, pad_mask)  # [B, T, embed_dim]

        if pooling:
            return self._mean_pool(x, ilens)
        return x


__all__ = ["MCConformerSpatialEncoder"]
