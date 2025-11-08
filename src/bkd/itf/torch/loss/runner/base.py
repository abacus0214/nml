"""Base class for torch loss runner."""

from torch import Tensor

from bkd.loss.runner.base import LossRunnerABC

__all__ = ["TorchLossRunner"]


class TorchLossRunner(LossRunnerABC[Tensor]):
    """Loss run for torch objects."""
