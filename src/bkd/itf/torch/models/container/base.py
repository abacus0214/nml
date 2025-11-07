"""Base torch Model child class."""

from torch import Tensor

from bkd.models.container.base import Model

__all__ = ["TorchModel"]


class TorchModel[InT, OutT, SampleT](Model[InT, OutT, SampleT, Tensor]):
    """Base torch Model child class."""
