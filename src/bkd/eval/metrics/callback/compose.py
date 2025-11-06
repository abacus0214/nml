"""Callbacks that compose/stack multiple callbkacs on top of each other."""

from bkd.eval.metrics.callback.abc import MetricCallback
from bkd.utils.typing.base.pydantic import RestrictedBaseModel
from bkd.utils.typing.eval.metrics import Metric, MetricID, Metrics
from bkd.utils.typing.events import EpochID

__all__ = ["CallbackCat"]


class CallbackCat(RestrictedBaseModel, MetricCallback):
    """Given a series of callbacks on construction, run all of them."""

    callbacks: tuple[MetricCallback]

    def log_epoch_start(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""
        for callback in self.callbacks:
            callback.log_epoch_start(eid=eid)

    def log_epoch_end(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""
        for callback in self.callbacks:
            callback.log_epoch_end(eid=eid)

    def log_metric(self, mid: MetricID, metric: Metric) -> None:
        """Log metric with given id."""
        for callback in self.callbacks:
            callback.log_metric(mid, metric)

    def log_metrics(self, metrics: Metrics) -> None:
        """Log multiple metrics at once, by default iterate."""
        for callback in self.callbacks:
            callback.log_metrics(metrics)

    def close(self) -> None:
        """Call at the end of the process."""
        for callback in self.callbacks:
            callback.close()
