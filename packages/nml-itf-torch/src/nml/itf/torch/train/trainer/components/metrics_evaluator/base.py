"""Base class for running evaluation on a split."""

from typing import Any, Iterable

from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.metrics_evaluator.base import MetricsEvaluator
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import Metrics
from nml.utils.typing.events import BatchID
from pydantic import Field
from torcheval.metrics import Metric as TorchEvalMetric

__all__ = ["TorchEvalEvaluator"]


class TorchEvalEvaluator(RestrictedBaseModel, MetricsEvaluator):
    """Base class for running evaluation on a split."""

    metrics: tuple[TorchEvalMetric[Any], ...] = Field(default_factory=tuple)

    def evaluate_batch[IpT, TgT](
        self,
        bid: BatchID,
        model: ModelABC[IpT, Any, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> Metrics:
        """Compute metrics ona given batch (could be training or validation)."""
        # Do inference once
        pred = model.forward(ipt=batch.ipt)

        # Evaluate on each metric
        for metric in self.metrics:
            # TODO: allow possibility to go through specific intepreter method
            # TODO: add possibility to customize target
            metric.update(pred, batch.tgt)
        return Metrics()

    def aggregate_f(
        self,
        metrics: Iterable[Metrics],
    ) -> Metrics:
        """Aggregate metrics from multiple batches."""
        # TODO: better strategy to extract name
        return Metrics(
            {
                str(metric.__class__.__name__): metric.compute()
                for metric in self.metrics
            }
        )
