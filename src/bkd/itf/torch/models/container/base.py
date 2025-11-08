"""Base torch Model child class."""

from torch import Tensor

from bkd.models.container.base import ModelABC

__all__ = ["TorchModel"]


class TorchModel[InT, OutT, SampleT](ModelABC[InT, OutT, SampleT, Tensor]):
    """Base torch Model child class."""
