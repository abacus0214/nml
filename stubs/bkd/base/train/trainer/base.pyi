import abc
from abc import ABC, abstractmethod
from bkd.base.callbacks.abc import EventCallback
from bkd.base.data.batches.container.base import BatchABC
from bkd.base.data.loader.container.base import DataLoaderABC
from bkd.base.loss.runner.base import LossRunnerABC
from bkd.base.models.container.base import ModelABC
from bkd.base.train.trainer.components.batch_loss.base import BatchLossTrainerABC
from bkd.base.train.trainer.components.split_evaluator.base import SplitEvaluator
from bkd.base.utils.loss.aggregators.base import LossAggregatorABC
from bkd.base.utils.typing.eval.metrics import MetricStep
from bkd.base.utils.typing.events import BatchID, EpochID
from typing import Any

__all__ = ['Trainer']

class TrainerABC[LossT](ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def train[IpT, TgT](self, model: ModelABC[IpT, Any, TgT, LossT], loss_runner: LossRunnerABC, dataset_splits: tuple[DataLoaderABC[Any, IpT, TgT], ...], callback: EventCallback = ...) -> None: ...

class Trainer[LossT](TrainerABC[LossT]):
    batch_loss_trainer: BatchLossTrainerABC[LossT]
    split_evaluator: SplitEvaluator
    loss_aggregator: LossAggregatorABC[LossT]
    num_epochs: int
    training_splits: tuple[bool, ...]
    loss_metric_name: str
    def is_training_split(self, split_idx: int) -> bool: ...
    def train[IpT, TgT](self, model: ModelABC[IpT, Any, TgT, LossT], loss_runner: LossRunnerABC, dataset_splits: tuple[DataLoaderABC[Any, IpT, TgT], ...], callback: EventCallback = ...) -> None: ...
    def epoch_step[IpT, TgT](self, eid: EpochID, model: ModelABC[IpT, Any, TgT, LossT], dataset: DataLoaderABC[Any, IpT, TgT], loss_runner: LossRunnerABC, update_model: bool = True, callback: EventCallback = ...) -> None: ...
    def epoch_loss[IpT, TgT](self, model: ModelABC[IpT, Any, TgT, LossT], dataset: DataLoaderABC[Any, IpT, TgT], loss_runner: LossRunnerABC, update_model: bool = False, callback: EventCallback = ...) -> float: ...
    def batch_step[IpT, TgT](self, bid: BatchID, model: ModelABC[IpT, Any, TgT, LossT], batch: BatchABC[Any, IpT, TgT], loss_runner: LossRunnerABC, update_model: bool = False, callback: EventCallback = ...) -> LossT: ...
    def evaluation[IpT, TgT](self, model: ModelABC[IpT, Any, TgT, Any], dataset_splits: tuple[DataLoaderABC[Any, IpT, TgT], ...], loss_runner: LossRunnerABC, step: None | MetricStep = None, callback: EventCallback = ...) -> None: ...
    def run_and_log_split_evaluation[IpT, TgT](self, model: ModelABC[IpT, Any, TgT, Any], dataset: DataLoaderABC[Any, IpT, TgT], step: None | MetricStep = None, callback: EventCallback = ...) -> None: ...
