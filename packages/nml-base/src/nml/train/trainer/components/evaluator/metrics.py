"""Evaluator parent class specific for scalar metrics."""

from typing import Any, Iterable

from nml.callbacks.abc import EventCallback
from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.evaluator.base import EvaluatorABC
from nml.utils.typing.eval.metrics import Metrics, MetricStep
from nml.utils.typing.events import BatchID, EpochID

__all__ = ["MetricsEvaluator"]


class MetricsEvaluator(EvaluatorABC[Metrics]):
    """Base class for running evaluation on a split."""

    def evaluate_batch[IpT, TgT](
        self,
        eid: EpochID,
        bid: BatchID,
        model: ModelABC[IpT, Any, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> Metrics:
        """Compute metrics ona given batch (could be training or validation)."""
        return Metrics()

    def aggregate_f(
        self,
        results: Iterable[Metrics],
    ) -> Metrics:
        """Aggregate metrics from multiple batches."""
        return Metrics()

    def log(
        self,
        step: MetricStep,
        results: Metrics,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Define how to log results."""
