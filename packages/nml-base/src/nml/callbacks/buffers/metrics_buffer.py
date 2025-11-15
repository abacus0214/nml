"""Callback that stores all observed metrics to be used later in the code."""

from nml.callbacks.stack import CallbackStack
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import (
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from pydantic import Field

__all__ = ["MetricsBufferCallback"]


class MetricsBufferCallback(RestrictedBaseModel, CallbackStack):
    """Record all events."""

    buffer: MetricsBatch = Field(default_factory=MetricsBatch)

    def log_metric_aux(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""
        self.log_metrics_aux({mid: metric}, step=step)

    def log_metrics_aux(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        self.log_metrics_batch_aux(MetricsBatch.from_metrics(metrics, step))

    def log_metrics_batch_aux(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        self.buffer.update(batch)
