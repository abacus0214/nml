"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod
from typing import Any, Mapping

from nml.callbacks.abc import EventCallback
from nml.data.batches.container.base import BatchABC
from nml.data.loader.container.base import DataLoaderABC
from nml.loss.runner.base import LossRunnerABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.batch_loss.base import BatchLossTrainerABC
from nml.train.trainer.components.evaluator.base import EvaluatorABC
from nml.utils.dicts.remap import add_prefix
from nml.utils.loss.aggregators.base import LossAggregatorABC
from nml.utils.loss.aggregators.mean import NPMeanAggregator
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.data.dataset import DatasetID
from nml.utils.typing.events import EpochID, TrainingStepID
from nml.utils.typing.logging.parameters import ParamsDict

__all__ = ["Trainer"]

DEFAULT_LOSS_METRIC_NAME = "loss"

type SplitsDict[BatchT, IpT, TgT] = dict[DatasetID, DataLoaderABC[BatchT, IpT, TgT]]


class TrainerABC[LossT](ABC):
    """Base class that defines the interface for a trainer."""

    @abstractmethod
    def train[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        train_splits: SplitsDict[Any, IpT, TgT],
        val_splits: None | SplitsDict[Any, IpT, TgT] = None,
        evaluators: tuple[EvaluatorABC[Any], ...] = tuple(),
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Update the given model."""

    @property
    @abstractmethod
    def params(self) -> ParamsDict:
        """Return parameters for this trainer in ParamsDict format."""


class Trainer[LossT](RestrictedBaseModel, TrainerABC[LossT]):
    """Pre made implementation that uses nml interfaces to do most of the work."""

    # Parameters
    batch_loss_trainer: BatchLossTrainerABC[LossT]

    loss_runner: LossRunnerABC
    loss_aggregator: LossAggregatorABC[LossT] = NPMeanAggregator[Any]()

    # Settings
    num_epochs: int

    # UI Settings
    loss_metric_name: str = DEFAULT_LOSS_METRIC_NAME

    @property
    def params(self) -> ParamsDict:
        """Return parameters for this trainer in ParamsDict format."""
        return {
            "num_epochs": self.num_epochs,
        }

    def train[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, LossT],
        train_splits: SplitsDict[Any, IpT, TgT],
        val_splits: None | SplitsDict[Any, IpT, TgT] = None,
        evaluators: tuple[EvaluatorABC[Any], ...] = tuple(),
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Perform training loop."""
        # Initialize callback
        with callback.start_context(params=self.params):
            # Log parameters
            callback.log_params(dict(add_prefix(self.params, "trainer")))
            callback.log_params(dict(add_prefix(model.params, "model")))

            # Perofrm a training epoch for `num_epochs` times
            for eid in range(self.num_epochs):
                # Trian on the entire epoch for each training split
                for split_id, train_split in train_splits.items():
                    self.epoch_step(
                        split_id=split_id,
                        eid=eid,
                        model=model,
                        dataset=train_split,
                        update_model=True,
                        evaluators=evaluators,
                        callback=callback,
                    )

                # Evaluate model
                self.evaluation(
                    eid=eid,
                    model=model,
                    val_splits=val_splits or {},
                    evaluators=evaluators,
                    callback=callback,
                )

    def epoch_step[IpT, TgT](
        self,
        split_id: DatasetID,
        eid: EpochID,
        model: ModelABC[IpT, Any, TgT, LossT],
        dataset: DataLoaderABC[Any, IpT, TgT],
        update_model: bool = True,
        evaluators: tuple[EvaluatorABC[Any], ...] = tuple(),
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Train for a single epoch."""
        # Communicate start of epoch
        callback.log_epoch_start(eid=eid, epoch_size=len(dataset))

        # Compute loss for each batch
        agg_loss, results = self.epoch_loss(
            split_id=split_id,
            eid=eid,
            model=model,
            dataset=dataset,
            update_model=update_model,
            evaluators=evaluators,
            callback=callback,
        )

        # Communicate aggregated loss across batches
        loss_metric_id = f"{split_id}/{self.loss_metric_name}"
        callback.log_metric(mid=loss_metric_id, metric=agg_loss, step=eid)
        # Log epoch metrics
        for result_dict, evaluator in zip(results, evaluators):
            evaluator.log(
                step=eid, results=add_prefix(result_dict, split_id), callback=callback
            )
        # Communicate epoch end
        callback.log_epoch_end(eid=eid)

    def epoch_loss[IpT, TgT](
        self,
        split_id: DatasetID,
        eid: EpochID,
        model: ModelABC[IpT, Any, TgT, LossT],
        dataset: DataLoaderABC[Any, IpT, TgT],
        update_model: bool = False,
        evaluators: tuple[EvaluatorABC[Any], ...] = tuple(),
        callback: EventCallback = EventCallback(),
    ) -> tuple[float, tuple[Mapping[str, Any], ...]]:
        """Compute the loss for a single epoch and update model (if specified)."""
        # Compute loss and metrics for each batch
        batches_results = [
            self.batch_step(
                step_id=TrainingStepID(
                    eid_max=self.num_epochs - 1, eid=eid, bid=bid, did=split_id
                ),
                model=model,
                batch=batch,
                update_model=update_model,
                evaluators=evaluators,
                callback=callback,
            )
            for bid, batch in dataset.epoch_iterator
        ]

        # Split losses and metrics
        losses, results_per_batch = zip(*batches_results)
        results_per_evaluator = zip(*results_per_batch)

        # Aggregate losses
        epoch_loss = self.loss_aggregator.aggregate_f(losses=losses)
        # Aggregate metrics
        epoch_metrics = tuple(
            (
                evaluator.aggregate_f(results=results_chunk)
                for results_chunk, evaluator in zip(results_per_evaluator, evaluators)
            )
        )

        return epoch_loss, epoch_metrics

    def batch_step[IpT, TgT](
        self,
        step_id: TrainingStepID,
        model: ModelABC[IpT, Any, TgT, LossT],
        batch: BatchABC[Any, IpT, TgT],
        update_model: bool = False,
        evaluators: tuple[EvaluatorABC[Any], ...] = tuple(),
        callback: EventCallback = EventCallback(),
    ) -> tuple[LossT, tuple[Mapping[str, Any], ...]]:
        """Train for a single batch."""
        # Communicate batch update start
        callback.log_batch_start(bid=step_id.bid)

        # Update model
        loss = self.batch_loss_trainer.batch_loss(
            loss_runner=self.loss_runner,
            model=model,
            batch=batch,
            update_model=update_model,
        )

        # Evaluate batch
        results = tuple(
            (
                evaluator.evaluate_batch(
                    step_id=step_id,
                    pred=model.inference(batch.ipt),
                    model=model,
                    batch=batch,
                )
                for evaluator in evaluators
            )
        )

        # Communicte batch update end
        callback.log_batch_end(bid=step_id.bid)
        return loss, results

    def evaluation[IpT, TgT](
        self,
        eid: EpochID,
        model: ModelABC[IpT, Any, TgT, Any],
        val_splits: SplitsDict[Any, IpT, TgT],
        evaluators: tuple[EvaluatorABC[Any], ...] = tuple(),
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Evaluate on each dataset."""
        # Go through all splits
        for split_id, dataset in val_splits.items():
            # Compute loss on each validation set
            val_loss, results = self.epoch_loss(
                split_id=split_id,
                eid=eid,
                model=model,
                dataset=dataset,
                update_model=False,
                evaluators=evaluators,
                callback=callback,
            )
            # Log validation loss
            callback.log_metric(
                mid=f"{split_id}/{self.loss_metric_name}",
                metric=val_loss,
                step=eid,
            )
            # Log metrics
            for result_dict, evaluator in zip(results, evaluators):
                evaluator.log(
                    step=eid,
                    results=add_prefix(result_dict, split_id),
                    callback=callback,
                )
