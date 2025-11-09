"""Base class for batch loss trainer."""

from abc import ABC, abstractmethod
from typing import Any

from bkd.callbacks.abc import EventCallback
from bkd.data.batches.container.base import BatchABC
from bkd.loss.runner.base import LossRunnerABC
from bkd.models.container.base import ModelABC

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
        callback: EventCallback = EventCallback(),
    ) -> LossT:
        """Compute loss and update model (if required)."""
