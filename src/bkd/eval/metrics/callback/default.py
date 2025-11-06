"""Default callbacks."""

from bkd.eval.metrics.callback.abc import MetricCallback
from bkd.utils.typing.eval.metrics import Metric, MetricID


class EmptyCallback(MetricCallback):
    """Simple callback that does nothing."""

    def log_metric(self, mid: MetricID, metric: Metric) -> None:
        """Log metric with given id."""
