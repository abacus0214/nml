"""Interface for evaluation callback."""

from bkd.utils.typing.eval.metrics import Metric, MetricID, Metrics
from bkd.utils.typing.events import BatchID, EpochID

__all__ = ["MetricCallback"]


class MetricCallback:
    """Callback to be executed when metric(s) is/are computed."""

    def start(self, num_epochs: None | int = None) -> None:
        """Call at the start of process."""

    def log_epoch_start(self, eid: EpochID, epoch_size: None | int = None) -> None:
        """Call this callback when epoch starts."""

    def log_epoch_end(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""

    def log_batch_start(self, bid: BatchID) -> None:
        """Call this callback when batch starts."""

    def log_batch_end(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""

    def log_metric(self, mid: MetricID, metric: Metric) -> None:
        """Log metric with given id."""

    def log_metrics(self, metrics: Metrics) -> None:
        """Log multiple metrics at once, by default iterate."""
        for mid, metric in metrics.items():
            self.log_metric(mid=mid, metric=metric)

    def close(self) -> None:
        """Call at the end of the process."""
