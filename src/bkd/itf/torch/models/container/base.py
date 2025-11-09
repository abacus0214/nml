"""Base torch Model child class."""

from typing import Hashable

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from bkd.models.container.base import ModelABC

__all__ = ["TorchModelABC"]


class TorchModelABC[InT, OutT, SampleT](ModelABC[InT, OutT, SampleT, Tensor]):
    """Base torch Model child class."""

    module: Module
    optimizer: None | Optimizer = None

    def forward(self, ipt: InT) -> OutT:
        """Run model prediction."""
        return self.module.forward(ipt)  # type: ignore

    @property
    def hash(self) -> Hashable:
        """Return hash for model."""
        return hash(str(self.module))
