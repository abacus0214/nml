"""Base class for loss runners."""

from abc import ABC, abstractmethod
from typing import Any

from bkd.data.batches.container.base import BatchABC
from bkd.models.container.base import ModelABC

__all__ = ["LossRunnerABC"]


class LossRunnerABC(ABC):
    """Wrapper for callacble that computes a loss based on a batch and model."""

    @abstractmethod
    def compute[IpT, TgT, LossT](
        self,
        batch: BatchABC[Any, IpT, TgT],
        model: ModelABC[IpT, Any, TgT, LossT],
    ) -> LossT:
        """Compute the loss from a given model and batch."""
