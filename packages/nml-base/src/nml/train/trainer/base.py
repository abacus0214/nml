"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod
from typing import Any, Mapping

from nml.callbacks.abc import EventCallback
from nml.data.batches.container.base import BatchABC
from nml.data.loader.container.base import DataLoaderABC
from nml.loss.runner.base import LossRunnerABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.batch_loss.base import BatchLossTrainerABC
from nml.train.trainer.components.evaluator.base import Evaluator, EvaluatorABC
from nml.utils.dicts.remap import add_prefix
from nml.utils.loss.aggregators.base import LossAggregatorABC
from nml.utils.loss.aggregators.mean import NPMeanAggregator
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.events import BatchID, EpochID

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
        evaluator: EvaluatorABC[Any] = Evaluator(),
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Update the given model."""


class Trainer[LossT](RestrictedBaseModel, TrainerABC[LossT]):
    """Pre made implementation that uses nml interfaces to do most of the work."""

    # Parameters
    batch_loss_trainer: BatchLossTrainerABC[LossT]

    loss_aggregator: LossAggregatorABC[LossT] = NPMeanAggregator[Any]()

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
        evaluator: EvaluatorABC[Any] = Evaluator(),
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
                        evaluator=evaluator,
                        callback=callback,
                    )

            # Evaluate model
            self.evaluation(
                eid=eid,
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
        evaluator: EvaluatorABC[Any] = Evaluator(),
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Train for a single epoch."""
        # Communicate start of epoch
        callback.log_epoch_start(eid=eid, epoch_size=len(dataset))

        # Compute loss for each batch
        agg_loss, results = self.epoch_loss(
            eid=eid,
            model=model,
            dataset=dataset,
            loss_runner=loss_runner,
            update_model=update_model,
            evaluator=evaluator,
            callback=callback,
        )

        # Communicate aggregated loss across batches
        loss_metric_id = f"{dataset.name}/{self.loss_metric_name}"
        callback.log_metric(mid=loss_metric_id, metric=agg_loss, step=eid)
        # Log epoch metrics
        evaluator.log(
            step=eid, results=add_prefix(results, dataset.name), callback=callback
        )
        # Communicate epoch end
        callback.log_epoch_end(eid=eid)

    def epoch_loss[IpT, TgT, ResultsT: Mapping[str, Any]](
        self,
        eid: EpochID,
        model: ModelABC[IpT, Any, TgT, LossT],
        dataset: DataLoaderABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        update_model: bool = False,
        # TODO: fix default value
        evaluator: EvaluatorABC[ResultsT] = Evaluator(),  # type: ignore
        callback: EventCallback = EventCallback(),
    ) -> tuple[float, ResultsT]:
        """Compute the loss for a single epoch and update model (if specified)."""
        # Compute loss and metrics for each batch
        batches_results = [
            self.batch_step(
                eid=eid,
                bid=bid,
                model=model,
                batch=batch,
                loss_runner=loss_runner,
                update_model=update_model,
                evaluator=evaluator,
                callback=callback,
            )
            for bid, batch in dataset.epoch_iterator
        ]

        # Split losses and metrics
        losses, results = zip(*batches_results)

        # Aggregate losses
        epoch_loss = self.loss_aggregator.aggregate_f(losses=losses)
        # Aggregate metrics
        epoch_metrics = evaluator.aggregate_f(results=results)

        return epoch_loss, epoch_metrics

    def batch_step[IpT, TgT, ResultsT: Mapping[str, Any]](
        self,
        eid: EpochID,
        bid: BatchID,
        model: ModelABC[IpT, Any, TgT, LossT],
        batch: BatchABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        update_model: bool = False,
        # TODO: fix default value
        evaluator: EvaluatorABC[ResultsT] = Evaluator(),  # type: ignore
        callback: EventCallback = EventCallback(),
    ) -> tuple[LossT, ResultsT]:
        """Train for a single batch."""
        # Communicate batch update start
        callback.log_batch_start(bid=bid)

        # Update model
        loss = self.batch_loss_trainer.batch_loss(
            loss_runner=loss_runner,
            model=model,
            batch=batch,
            update_model=update_model,
        )

        # Evaluate batch
        results = evaluator.evaluate_batch(
            eid_max=self.num_epochs - 1,
            eid=eid,
            bid=bid,
            pred=model.inference(batch.ipt),
            model=model,
            batch=batch,
        )

        # Communicte batch update end
        callback.log_batch_end(bid=bid)
        return loss, results

    def evaluation[IpT, TgT](
        self,
        eid: EpochID,
        model: ModelABC[IpT, Any, TgT, Any],
        dataset_splits: tuple[DataLoaderABC[Any, IpT, TgT], ...],
        loss_runner: LossRunnerABC,
        evaluator: EvaluatorABC[Any] = Evaluator(),
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Evaluate on each dataset."""
        # Go through all splits
        for split_idx, dataset in enumerate(dataset_splits):
            if not self.is_training_split(split_idx):
                # Compute loss on each validation set
                val_loss, results = self.epoch_loss(
                    eid=eid,
                    model=model,
                    dataset=dataset,
                    loss_runner=loss_runner,
                    update_model=False,
                    evaluator=evaluator,
                    callback=callback,
                )
                # Log validation loss
                callback.log_metric(
                    mid=f"{dataset.name}/{self.loss_metric_name}",
                    metric=val_loss,
                    step=eid,
                )
                # Log metrics
                evaluator.log(
                    step=eid,
                    results=add_prefix(results, dataset.name),
                    callback=callback,
                )
