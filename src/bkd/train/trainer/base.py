"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod
from typing import Any

from bkd.callbacks.abc import EventCallback
from bkd.data.batches.container.base import BatchABC
from bkd.data.loader.container.base import DataLoaderABC
from bkd.loss.runner.base import LossRunnerABC
from bkd.models.container.base import ModelABC
from bkd.utils.typing.eval.metrics import Metrics, MetricStep
from bkd.utils.typing.events import BatchID, EpochID

__all__ = ["Trainer", "DEFAULT_TRAIN_SPLIT_NAME", "DEFAULT_VAL_SPLIT_NAME"]

DEFAULT_TRAIN_SPLIT_NAME = "train"
DEFAULT_VAL_SPLIT_NAME = "val"


class Trainer[LossT](ABC):
    """Base class that defines the interface for a trainer."""

    num_epochs: int

    train_split_name: str = DEFAULT_TRAIN_SPLIT_NAME
    val_split_name: str = DEFAULT_VAL_SPLIT_NAME

    def train[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        train_set: DataLoaderABC[Any, IpT, TgT],
        val_set: DataLoaderABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC[LossT],
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Perform training loop."""
        # Initialize callback
        callback.start(num_epochs=self.num_epochs)

        # Perofrm a training epoch for `num_epochs` times
        for eid in range(self.num_epochs):
            # Trian on the entire epoch
            self.train_epoch(
                eid=eid,
                model=model,
                dataset=train_set,
                loss_runner=loss_runner,
                callback=callback,
            )
            # Evaluate model
            self.run_evaluation(
                step=eid,
                model=model,
                train_set=train_set,
                val_set=val_set,
                callback=callback,
            )

        callback.close()

    @abstractmethod
    def train_epoch[IpT, TgT](
        self,
        eid: EpochID,
        model: ModelABC[IpT, Any, TgT, LossT],
        dataset: DataLoaderABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC[LossT],
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Train for a single epoch."""
        callback.log_epoch_start(eid=eid)
        for bid, batch in enumerate(dataset):
            self.single_batch_update(
                bid=bid,
                model=model,
                batch=batch,
                loss_runner=loss_runner,
                callback=callback,
            )
        callback.log_epoch_end(eid=eid)

    def single_batch_update[IpT, TgT](
        self,
        bid: BatchID,
        model: ModelABC[IpT, Any, TgT, LossT],
        batch: BatchABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC[LossT],
        callback: EventCallback = EventCallback(),
    ) -> LossT:
        """Train for a single batch."""
        callback.log_batch_start(bid=bid)

        # Compute loss on batch
        loss = loss_runner.compute(batch=batch, model=model)
        # Update model
        self.update_from_loss(loss=loss)

        # Update model from loss
        callback.log_batch_end(bid=bid)
        return loss

    @abstractmethod
    def update_from_loss(self, loss: LossT) -> None:
        """Update model from loss."""

    def run_evaluation[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        train_set: DataLoaderABC[Any, IpT, TgT],
        val_set: DataLoaderABC[Any, IpT, TgT],
        step: None | MetricStep = None,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Evaluate on trian and validation sets."""
        self.run_split_evaluation(
            model=model,
            dataset=train_set,
            split_name=self.train_split_name,
            step=step,
            callback=callback,
        )
        self.run_split_evaluation(
            model=model,
            dataset=val_set,
            split_name=self.val_split_name,
            step=step,
            callback=callback,
        )

    def run_split_evaluation[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        dataset: DataLoaderABC[Any, IpT, TgT],
        split_name: None | str = None,
        step: None | MetricStep = None,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Compute metrics and log them.

        This method also sanitizes metrics names by adding a suffix, useful when calling
        this method for multiple splits.
        """
        return callback.log_metrics(
            {
                (mid if split_name is None else f"{split_name}/{mid}"): metric
                for mid, metric in self.evaluate_split(
                    model=model, dataset=dataset
                ).items()
            },
            step=step,
        )

    def evaluate_split[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        dataset: DataLoaderABC[Any, IpT, TgT],
    ) -> Metrics:
        """Compute metrics ona given set (could be training or validation)."""
        return {}
