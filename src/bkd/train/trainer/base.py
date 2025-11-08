"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import numpy as np

from bkd.callbacks.abc import EventCallback
from bkd.data.batches.container.base import BatchABC
from bkd.data.loader.container.base import DataLoaderABC
from bkd.loss.runner.base import LossRunnerABC
from bkd.models.container.base import ModelABC
from bkd.utils.typing.eval.metrics import Metrics, MetricStep
from bkd.utils.typing.events import BatchID, EpochID

__all__ = ["Trainer", "DEFAULT_TRAIN_SPLIT_NAME", "DEFAULT_VAL_SPLIT_NAME"]

DEFAULT_LOSS_METRIC_NAME = "loss"

DEFAULT_TRAIN_SPLIT_NAME = "train"
DEFAULT_VAL_SPLIT_NAME = "val"


class Trainer[LossT](ABC):
    """Base class that defines the interface for a trainer."""

    # Parameters
    num_epochs: int

    # TODO: fix the type ignore
    loss_aggregator: Callable[[Iterable[float]], float] = np.mean  # type: ignore

    # Settings
    train_split_name: str = DEFAULT_TRAIN_SPLIT_NAME
    val_split_name: str = DEFAULT_VAL_SPLIT_NAME
    loss_metric_name: str = DEFAULT_LOSS_METRIC_NAME

    def train[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        train_set: DataLoaderABC[Any, IpT, TgT],
        val_set: DataLoaderABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Perform training loop."""
        # Initialize callback
        callback.start(num_epochs=self.num_epochs)

        # Perofrm a training epoch for `num_epochs` times
        for eid in range(self.num_epochs):
            # Trian on the entire epoch
            self.epoch_step(
                eid=eid,
                model=model,
                dataset=train_set,
                loss_runner=loss_runner,
                update_model=True,
                split_name=self.train_split_name,
                callback=callback,
            )
            # Evaluate model
            self.evaluation(
                step=eid,
                model=model,
                train_set=train_set,
                val_set=val_set,
                loss_runner=loss_runner,
                callback=callback,
            )

        callback.close()

    def epoch_step[IpT, TgT](
        self,
        eid: EpochID,
        model: ModelABC[IpT, Any, TgT, LossT],
        dataset: DataLoaderABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        update_model: bool = True,
        split_name: None | str = None,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Train for a single epoch."""
        # Communicate start of epoch
        callback.log_epoch_start(eid=eid)

        # Compute loss for each batch
        agg_loss = self.epoch_loss(
            model=model,
            dataset=dataset,
            loss_runner=loss_runner,
            update_model=update_model,
            callback=callback,
        )
        # Compute name of metric to log loss
        loss_metrics_id = (
            self.loss_metric_name
            if split_name is None
            else f"{split_name}/{self.loss_metric_name}"
        )
        # Communicate aggregate losses across batches
        callback.log_metric(mid=loss_metrics_id, metric=agg_loss)
        # Communicate epoch end
        callback.log_epoch_end(eid=eid)

    def epoch_loss[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        dataset: DataLoaderABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        update_model: bool = False,
        callback: EventCallback = EventCallback(),
    ) -> float:
        """Compute the loss for a single epoch and update model (if specified)."""
        # Compute loss for each batch
        return self.loss_aggregator(
            [
                model.likelihood_to_float(
                    self.batch_step(
                        bid=bid,
                        model=model,
                        batch=batch,
                        loss_runner=loss_runner,
                        update_model=update_model,
                        callback=callback,
                    )
                )
                for bid, batch in enumerate(dataset)
            ]
        )

    def batch_step[IpT, TgT](
        self,
        bid: BatchID,
        model: ModelABC[IpT, Any, TgT, LossT],
        batch: BatchABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        update_model: bool = False,
        callback: EventCallback = EventCallback(),
    ) -> LossT:
        """Train for a single batch."""
        # Communicate batch update start
        callback.log_batch_start(bid=bid)

        # Update model
        loss = self.batch_loss(
            loss_runner=loss_runner,
            model=model,
            batch=batch,
            update_model=update_model,
            callback=callback,
        )

        # Communicte batch update end
        callback.log_batch_end(bid=bid)
        return loss

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

    def evaluation[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        train_set: DataLoaderABC[Any, IpT, TgT],
        val_set: DataLoaderABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        step: None | MetricStep = None,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Evaluate on trian and validation sets."""
        self.run_and_log_split_evaluation(
            model=model,
            dataset=train_set,
            split_name=self.train_split_name,
            step=step,
            callback=callback,
        )
        self.run_and_log_split_evaluation(
            model=model,
            dataset=val_set,
            split_name=self.val_split_name,
            step=step,
            callback=callback,
        )

        # Compute loss on validation set
        val_loss = self.epoch_loss(
            model=model,
            dataset=val_set,
            loss_runner=loss_runner,
            update_model=False,
            callback=callback,
        )
        # Log validation loss
        callback.log_metric(
            mid=f"{self.val_split_name}/{self.loss_metric_name}", metric=val_loss
        )

    def run_and_log_split_evaluation[IpT, TgT](
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
