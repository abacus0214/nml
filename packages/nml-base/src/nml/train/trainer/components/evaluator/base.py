"""Base class for running evaluation on a split or batch."""

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping

from nml.callbacks.abc import EventCallback
from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.utils.typing.eval.metrics import MetricStep
from nml.utils.typing.events import TrainingStepID

__all__ = ["EvaluatorABC", "Evaluator"]


class EvaluatorABC[ResultsT: Mapping[str, Any]](ABC):
    """Parent class for evaluators."""

    @abstractmethod
    # TODO: pass prediction/model output so that it does not have to be repeated
    def evaluate_batch[IpT, OutT, TgT](
        self,
        step_id: TrainingStepID,
        pred: OutT,
        model: ModelABC[IpT, OutT, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> ResultsT:
        """Compute results ona given batch (could be training or validation)."""

    @abstractmethod
    def aggregate_f(
        self,
        results: Iterable[ResultsT],
    ) -> ResultsT:
        """Aggregate results from multiple batches."""

    @abstractmethod
    def log(
        self,
        step: MetricStep,
        results: ResultsT,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Define how to log results."""


class Evaluator(EvaluatorABC[dict[str, Any]]):
    """Default empty evaluator."""

    def evaluate_batch[IpT, OutT, TgT](
        self,
        step_id: TrainingStepID,
        pred: OutT,
        model: ModelABC[IpT, OutT, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> dict[str, Any]:
        """Compute results ona given batch (could be training or validation)."""
        return {}

    def aggregate_f(
        self,
        results: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate results from multiple batches."""
        return {}

    def log(
        self,
        step: MetricStep,
        results: dict[str, Any],
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Define how to log results."""
