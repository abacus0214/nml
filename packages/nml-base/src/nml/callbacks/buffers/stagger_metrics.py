"""Stack callbacks that lags metrics logging."""

from types import TracebackType

from nml.callbacks.stack import CallbackStack
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import (
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from nml.utils.typing.logging.parameters import ParamsDict
from pydantic import Field


class StaggerMetricsStack(RestrictedBaseModel, CallbackStack):
    """buffer all metrics before forwarding them to the sub callback."""

    # Settings
    buffer_max_size: int = 10

    # State
    buffer: MetricsBatch = Field(init=False, default_factory=MetricsBatch)

    def check_buffer_flush(self) -> None:
        """Check if buffer needs to be flushed, if so do so."""
        if len(self.buffer) >= self.buffer_max_size:
            self.flush_buffer()

    def flush_buffer(self) -> None:
        """Log all metrics in the buffer and clear."""
        if len(self.buffer) > 0:
            self.callback.log_metrics_batch(self.buffer)
            self.clear_buffer()

    def clear_buffer(self) -> None:
        """Delete the contents of the buffer."""
        self.buffer.clear()

    def start_aux(self, params: None | ParamsDict = None) -> None:
        """Call at the start of process."""
        self.clear_buffer()

    def log_metric(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""
        self.log_metrics(Metrics({mid: metric}), step=step)

    def log_metrics(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        self.log_metrics_batch(MetricsBatch.from_metrics(metrics, step))

    def log_metrics_batch(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        self.buffer.update(batch)
        self.check_buffer_flush()

    def close_aux(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None | bool:
        """Call at the end of the process."""
        self.flush_buffer()
        return False
