"""Callbacks that compose/stack multiple callbkacs on top of each other."""

from types import TracebackType
from typing import Any

from nml.callbacks.abc import EventCallback
from nml.models.container.base import ModelABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import (
    Figure,
    FigureID,
    Figures,
    FiguresBatch,
    FigureStep,
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from nml.utils.typing.events import BatchID, EpochID
from nml.utils.typing.models.container.base import ModelID

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

    def log_model(
        self,
        model: ModelABC[Any, Any, Any, Any],
        name: None | ModelID = None,
        step: MetricStep = None,
    ) -> None:
        """Log a model."""
        for callback in self.callbacks:
            callback.log_model(model=model, name=name, step=step)

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

    def log_figure(
        self, fid: FigureID, figure: Figure, step: FigureStep = None
    ) -> None:
        """Log figure with given id."""
        for callback in self.callbacks:
            callback.log_figure(fid, figure, step=step)

    def log_figures(self, figures: Figures, step: FigureStep = None) -> None:
        """Log multiple figures at once, by default iterate."""
        for callback in self.callbacks:
            callback.log_figures(figures, step=step)

    def log_figure_batch(self, batch: FiguresBatch) -> None:
        """Log multiple figures samples at arbitrary timesteps at once, by default iterate."""
        for callback in self.callbacks:
            callback.log_figure_batch(batch)

    def close(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None | bool:
        """Call at the end of the process."""
        return any(
            (callback.close(exc_type, exc_val, exc_tb) for callback in self.callbacks)
        )
