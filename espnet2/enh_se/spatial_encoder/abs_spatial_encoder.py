from abc import ABC, abstractmethod
from typing import Optional

import torch
from typeguard import typechecked


class AbsSpatialEncoder(torch.nn.Module, ABC):
    """Abstract base class for spatial encoders"""
    
    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        num_channels: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward.
        
        Args:
            input: [B, T, F] or [B, T, C, F] - STFT spectrum (complex)
            ilens: [B] - input lengths
            num_channels: Number of input channels (1 for SC, num_channels_mc for MC).
                         If None, automatically determined from input shape.
        Returns:
            embedding: [B, embed_dim] - spatial embedding
        """
        raise NotImplementedError

