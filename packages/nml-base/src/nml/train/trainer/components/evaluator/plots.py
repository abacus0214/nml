"""Parent class for plots evaluator."""

from typing import Any, Iterable

from nml.callbacks.abc import EventCallback
from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.evaluator.base import EvaluatorABC
from nml.utils.typing.eval.metrics import Figures, MetricStep
from nml.utils.typing.events import BatchID, EpochID

__all__ = ["PlotsEvaluator"]


class PlotsEvaluator(EvaluatorABC[Figures]):
    """Baseclass for generating plots on a split."""

    def evaluate_batch[IpT, OutT, TgT](
        self,
        eid: EpochID,
        bid: BatchID,
        pred: OutT,
        model: ModelABC[IpT, OutT, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> Figures:
        """Compute results ona given batch (could be training or validation)."""
        return Figures()

    def aggregate_f(
        self,
        results: Iterable[Figures],
    ) -> Figures:
        """Aggregate results from multiple batches."""
        return Figures()

    def log(
        self,
        step: MetricStep,
        results: Figures,
        callback: EventCallback = EventCallback(),
    ) -> None:
        """Define how to log results."""
