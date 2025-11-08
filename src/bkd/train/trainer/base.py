"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod
from typing import Any

from bkd.callbacks.abc import EventCallback
from bkd.data.container.base import DataLoader
from bkd.models.container.base import Model
from bkd.models.loss.runner.base import LossRunner
from bkd.utils.typing.events import EpochID

__all__ = ["Trainer"]


class Trainer[ModelT: Model[Any, Any, Any, Any], BatchT, LossT: LossRunner[Any]](ABC):
    """Base class that defines the interface for a trainer."""

    num_epochs: int

    def train(
        self,
        model: ModelT,
        dataset: DataLoader[BatchT],
        loss_runner: LossT,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Perform training loop."""
        # Initialize callback
        callback.start(num_epochs=self.num_epochs)

        # Perofrm a training epoch for `num_epochs` times
        for eid in range(self.num_epochs):
            callback.log_epoch_start(eid=eid)
            self.train_epoch(
                eid=eid,
                model=model,
                dataset=dataset,
                loss_runner=loss_runner,
                callback=callback,
            )
            callback.log_epoch_end(eid=eid)

        callback.close()

    @abstractmethod
    def train_epoch(
        self,
        eid: EpochID,
        model: ModelT,
        dataset: DataLoader[BatchT],
        loss_runner: LossT,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Train for a single epoch."""
