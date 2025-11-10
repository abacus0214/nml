"""Base class for batch loss trainer."""

from abc import ABC, abstractmethod
from typing import Any

from bkd.base.data.batches.container.base import BatchABC
from bkd.base.loss.runner.base import LossRunnerABC
from bkd.base.models.container.base import ModelABC

__all__ = ["BatchLossTrainerABC"]


class BatchLossTrainerABC[LossT](ABC):
    """Base class for batch loss trainer."""

    @abstractmethod
    def batch_loss[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        batch: BatchABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        update_model: bool = False,
    ) -> LossT:
        """Compute loss and update model (if required)."""
