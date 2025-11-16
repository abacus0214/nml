"""Base class for running evaluation on a split or batch."""

from typing import Any, Iterable

from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.utils.typing.eval.metrics import Metrics
from nml.utils.typing.events import BatchID

__all__ = ["MetricsEvaluator"]


class MetricsEvaluator:
    """Base class for running evaluation on a split."""

    def evaluate_batch[IpT, TgT](
        self,
        bid: BatchID,
        model: ModelABC[IpT, Any, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> Metrics:
        """Compute metrics ona given batch (could be training or validation)."""
        return Metrics()

    def aggregate_f(
        self,
        metrics: Iterable[Metrics],
    ) -> Metrics:
        """Aggregate metrics from multiple batches."""
        return Metrics()
