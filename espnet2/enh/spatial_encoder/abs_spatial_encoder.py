from abc import ABC, abstractmethod
import torch
from typeguard import typechecked


class AbsSpatialEncoder(torch.nn.Module, ABC):
    """Abstract base class for spatial encoders"""
    
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward.
        
        Args:
            input: [B, T] or [B, C, T] - waveform
        Returns:
            embedding: [B, embed_dim] - spatial embedding
        """
        raise NotImplementedError

