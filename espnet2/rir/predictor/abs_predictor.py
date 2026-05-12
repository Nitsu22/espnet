from abc import ABC, abstractmethod
from typing import List

import torch


class AbsRIRPredictor(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_sources(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_mics(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def rir_length(self) -> int:
        raise NotImplementedError
