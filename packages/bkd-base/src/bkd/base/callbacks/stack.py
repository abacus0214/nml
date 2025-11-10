"""Useful interface for stacking callbacks."""

from bkd.callbacks.abc import EventCallback
from bkd.utils.typing.eval.metrics import (
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from bkd.utils.typing.events import BatchID, EpochID

__all__ = ["CallbackStack"]


class CallbackStack(EventCallback):
    """Stack callback on top of each other.

    Intercept or modify events sent to inner callback.
    """

    callback: EventCallback = EventCallback()

    def start(self, num_epochs: None | int = None) -> None:
        """Call at the start of the process."""
        self.start_aux(num_epochs=num_epochs)
        self.callback.start(num_epochs=num_epochs)

    def start_aux(self, num_epochs: None | int = None) -> None:
        """Call at the start of the process."""

    def log_epoch_start(self, eid: EpochID, epoch_size: None | int = None) -> None:
        """Call this callback when epoch ends."""
        self.log_epoch_start_aux(eid=eid, epoch_size=epoch_size)
        self.callback.log_epoch_start(eid=eid, epoch_size=epoch_size)

    def log_epoch_start_aux(self, eid: EpochID, epoch_size: None | int = None) -> None:
        """Call this callback when epoch ends."""

    def log_epoch_end(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""
        self.log_epoch_end_aux(eid=eid)
        self.callback.log_epoch_end(eid=eid)

    def log_epoch_end_aux(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""

    def log_batch_start(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""
        self.log_batch_start_aux(bid=bid)
        self.callback.log_batch_start(bid=bid)

    def log_batch_start_aux(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""

    def log_batch_end(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""
        self.log_batch_end_aux(bid=bid)
        self.callback.log_batch_end(bid=bid)

    def log_batch_end_aux(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""

    def log_metric(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""
        self.log_metric_aux(mid=mid, metric=metric, step=step)
        self.callback.log_metric(mid=mid, metric=metric, step=step)

    def log_metric_aux(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""

    def log_metrics(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        self.log_metrics_aux(metrics=metrics, step=step)
        self.callback.log_metrics(metrics=metrics, step=step)

    def log_metrics_aux(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""

    def log_metrics_batch(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        self.log_batch_aux(batch=batch)
        self.callback.log_metrics_batch(batch=batch)

    def log_batch_aux(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""

    def close(self) -> None:
        """Call at the end of the process."""
        self.close_aux()
        self.callback.close()

    def close_aux(self) -> None:
        """Call at the end of the process."""
