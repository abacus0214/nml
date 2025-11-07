"""Base class for torch loss runner."""

from torch import Tensor

from bkd.models.loss.runner.base import LossRunner

__all__ = ["TorchLossRunner"]


class TorchLossRunner(LossRunner[Tensor]):
    """Loss run for torch objects."""
