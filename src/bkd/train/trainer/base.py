"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod
from typing import Any

from bkd.callbacks.abc import EventCallback
from bkd.data.batches.container.base import BatchABC
from bkd.data.loader.container.base import DataLoaderABC
from bkd.loss.runner.base import LossRunnerABC
from bkd.models.container.base import ModelABC
from bkd.train.trainer.components.batch_loss.base import BatchLossTrainerABC
from bkd.train.trainer.components.split_evaluator.base import SplitEvaluator
from bkd.utils.loss.aggregators.base import LossAggregatorABC
from bkd.utils.loss.aggregators.mean import NPMeanAggregator
from bkd.utils.typing.eval.metrics import MetricStep
from bkd.utils.typing.events import BatchID, EpochID

__all__ = ["Trainer"]

DEFAULT_LOSS_METRIC_NAME = "loss"


class TrainerABC[LossT](ABC):
    """Base class that defines the interface for a trainer."""

    @abstractmethod
    def train[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        loss_runner: LossRunnerABC,
        dataset_splits: tuple[DataLoaderABC[Any, IpT, TgT], ...],
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Update the given model."""


class Trainer[LossT](TrainerABC[LossT]):
    """Pre made implementation that uses bkd interfaces to do most of the work."""

    # Parameters
    batch_loss_trainer: BatchLossTrainerABC[LossT]

    split_evaluator: SplitEvaluator = SplitEvaluator()
    loss_aggregator: LossAggregatorABC[float] = NPMeanAggregator[float]()

    # Settings
    num_epochs: int
    training_splits: tuple[bool, ...] = (True,)

    # UI Settings
    loss_metric_name: str = DEFAULT_LOSS_METRIC_NAME

    def is_training_split(self, split_idx: int) -> bool:
        """Return true if split is training split."""
        return split_idx < len(self.training_splits) and self.training_splits[split_idx]

    def train[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        loss_runner: LossRunnerABC,
        dataset_splits: tuple[DataLoaderABC[Any, IpT, TgT], ...],
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Perform training loop."""
        # Initialize callback
        callback.start(num_epochs=self.num_epochs)

        # Perofrm a training epoch for `num_epochs` times
        for eid in range(self.num_epochs):
            # Trian on the entire epoch for each training split
            for split_idx, train_split in enumerate(dataset_splits):
                if self.is_training_split(split_idx):
                    self.epoch_step(
                        eid=eid,
                        model=model,
                        dataset=train_split,
                        loss_runner=loss_runner,
                        update_model=True,
                        callback=callback,
                    )

            # Evaluate model
            self.evaluation(
                step=eid,
                model=model,
                dataset_splits=dataset_splits,
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

        # Communicate aggregated loss across batches
        loss_metric_id = f"{dataset.name}/{self.loss_metric_name}"
        callback.log_metric(mid=loss_metric_id, metric=agg_loss)
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
        loss = self.batch_loss_trainer.batch_loss(
            loss_runner=loss_runner,
            model=model,
            batch=batch,
            update_model=update_model,
            callback=callback,
        )

        # Communicte batch update end
        callback.log_batch_end(bid=bid)
        return loss

    def evaluation[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        dataset_splits: tuple[DataLoaderABC[Any, IpT, TgT], ...],
        loss_runner: LossRunnerABC,
        step: None | MetricStep = None,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Evaluate on each dataset."""
        # Go through all splits
        for split_idx, dataset in enumerate(dataset_splits):
            # Compute metrics
            self.run_and_log_split_evaluation(
                model=model,
                dataset=dataset,
                step=step,
                callback=callback,
            )

            # Compute loss on each validation set
            if not self.is_training_split(split_idx):
                val_loss = self.epoch_loss(
                    model=model,
                    dataset=dataset,
                    loss_runner=loss_runner,
                    update_model=False,
                    callback=callback,
                )
                # Log validation loss
                callback.log_metric(
                    mid=f"{dataset.name}/{self.loss_metric_name}", metric=val_loss
                )

    def run_and_log_split_evaluation[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        dataset: DataLoaderABC[Any, IpT, TgT],
        step: None | MetricStep = None,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Compute metrics and log them.

        This method also sanitizes metrics names by adding a suffix, useful when calling
        this method for multiple splits.
        """
        return callback.log_metrics(
            {
                f"{dataset.name}/{mid}": metric
                for mid, metric in self.split_evaluator.evaluate_split(
                    model=model, dataset=dataset
                ).items()
            },
            step=step,
        )
