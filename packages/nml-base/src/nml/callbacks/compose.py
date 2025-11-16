"""Callbacks that compose/stack multiple callbkacs on top of each other."""

from nml.callbacks.abc import EventCallback
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import (
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from nml.utils.typing.events import BatchID, EpochID

__all__ = ["CallbackCat"]


class CallbackCat(RestrictedBaseModel, EventCallback):
    """Given a series of callbacks on construction, run all of them."""

    callbacks: tuple[EventCallback, ...]

    def start(self, num_epochs: None | int = None) -> None:
        """Forward start of process."""
        for callback in self.callbacks:
            callback.start(num_epochs=num_epochs)

    def log_epoch_start(self, eid: EpochID, epoch_size: None | int = None) -> None:
        """Call this callback when epoch ends."""
        for callback in self.callbacks:
            callback.log_epoch_start(eid=eid, epoch_size=epoch_size)

    def log_epoch_end(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""
        for callback in self.callbacks:
            callback.log_epoch_end(eid=eid)

    def log_batch_start(self, bid: BatchID) -> None:
        """Call this callback when batch starts."""
        for callback in self.callbacks:
            callback.log_batch_start(bid=bid)

    def log_batch_end(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""
        for callback in self.callbacks:
            callback.log_batch_end(bid=bid)

    def log_metric(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""
        for callback in self.callbacks:
            callback.log_metric(mid, metric, step=step)

    def log_metrics(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        for callback in self.callbacks:
            callback.log_metrics(metrics, step=step)

    def log_metrics_batch(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        for callback in self.callbacks:
            callback.log_metrics_batch(batch)

    def close(self) -> None:
        """Call at the end of the process."""
        for callback in self.callbacks:
            callback.close()
