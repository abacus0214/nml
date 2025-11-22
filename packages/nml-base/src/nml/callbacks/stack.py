"""Useful interface for stacking callbacks."""

from typing import Any

from nml.callbacks.abc import EventCallback
from nml.models.container.base import ModelABC
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

    def log_model(
        self, model: ModelABC[Any, Any, Any, Any], step: MetricStep = None
    ) -> None:
        """Log a model."""
        self.log_model_aux(model=model, step=step)
        self.callback.log_model(model=model, step=step)

    def log_model_aux(
        self, model: ModelABC[Any, Any, Any, Any], step: MetricStep = None
    ) -> None:
        """Log a model."""

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
        self.log_metrics_batch_aux(batch=batch)
        self.callback.log_metrics_batch(batch=batch)

    def log_metrics_batch_aux(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""

    def log_figure(
        self, fid: FigureID, figure: Figure, step: FigureStep = None
    ) -> None:
        """Log figure with given id."""
        self.log_figure_aux(fid=fid, figure=figure, step=step)
        self.callback.log_figure(fid=fid, figure=figure, step=step)

    def log_figure_aux(
        self, fid: FigureID, figure: Figure, step: FigureStep = None
    ) -> None:
        """Log figure with given id."""

    def log_figures(self, figures: Figures, step: FigureStep = None) -> None:
        """Log multiple figures at once, by default iterate."""
        self.log_figures_aux(figures=figures, step=step)
        self.callback.log_figures(figures=figures, step=step)

    def log_figures_aux(self, figures: Figures, step: FigureStep = None) -> None:
        """Log multiple figures at once, by default iterate."""

    def log_figure_batch(self, batch: FiguresBatch) -> None:
        """Log multiple figures samples at arbitrary timesteps at once, by default iterate."""
        self.log_figure_batch_aux(batch=batch)
        self.callback.log_figure_batch(batch=batch)

    def log_figure_batch_aux(self, batch: FiguresBatch) -> None:
        """Log multiple figures samples at arbitrary timesteps at once, by default iterate."""

    def close(self) -> None:
        """Call at the end of the process."""
        self.close_aux()
        self.callback.close()

    def close_aux(self) -> None:
        """Call at the end of the process."""
