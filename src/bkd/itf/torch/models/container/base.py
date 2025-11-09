"""Base torch Model child class."""

from abc import abstractmethod

from torch import Tensor
from torch.optim import Optimizer

from bkd.models.container.base import ModelABC

__all__ = ["TorchModelABC"]


class TorchModelABC[InT, OutT, SampleT](ModelABC[InT, OutT, SampleT, Tensor]):
    """Base torch Model child class."""

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer:
        """Get optimizer for this model."""
