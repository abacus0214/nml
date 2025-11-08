"""Base class for loss runners."""

from abc import ABC, abstractmethod

from bkd.data.batches.container.base import BatchABC
from bkd.models.container.base import ModelABC

__all__ = ["LossRunnerABC"]


class LossRunnerABC[LossT](ABC):
    """Wrapper for callacble that computes a loss based on a batch and model."""

    @abstractmethod
    def compute[IpT, TgT](
        self,
        batch: BatchABC[object, IpT, TgT],
        model: ModelABC[IpT, object, TgT, LossT],
    ) -> LossT:
        """Compute the loss from a given model and batch."""
