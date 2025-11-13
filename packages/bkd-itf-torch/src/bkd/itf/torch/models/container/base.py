"""Base torch Model child class."""

from typing import Hashable

from bkd.models.container.base import ModelABC
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

__all__ = ["TorchModelABC"]


class TorchModelABC[InT, OutT, SampleT](ModelABC[InT, OutT, SampleT, Tensor]):
    """Base torch Model child class. Easy containment of torch.nn.Module.

    This class allows to wrap a torch.nn.Module as a bkd Model. Also allows the
    user to specify an optimizer to define a custom optimizer to be used for
    this model (some optimizer maps can use this to override default behavior).
    """

    module: Module
    optimizer: None | Optimizer = None

    def forward(self, ipt: InT) -> OutT:
        """Run model prediction."""
        return self.module.forward(ipt)  # type: ignore

    @property
    def hash(self) -> Hashable:
        """Return hash for model."""
        return hash(str(self.module))
