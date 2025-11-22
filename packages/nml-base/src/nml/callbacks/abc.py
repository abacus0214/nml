"""Interface for evaluation callback."""

from types import TracebackType
from typing import Any

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

__all__ = ["EventCallback"]


class EventCallback:
    """Callback to be executed when metric(s) is/are computed."""

    def __enter__(self) -> None:
        """Callbacks are context managers.

        This is done to be able to catch exceptions within the context.
        """

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None | bool:
        """Close the context."""
        return self.close(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

    def start_context(self, num_epochs: None | int = None) -> "EventCallback":
        """Call at the start of process while creating context."""
        self.start(num_epochs=num_epochs)
        return self

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

    def log_model(
        self, model: ModelABC[Any, Any, Any, Any], step: MetricStep = None
    ) -> None:
        """Log a model."""

    def log_metric(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""

    def log_metrics(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        for mid, metric in metrics.items():
            self.log_metric(mid=mid, metric=metric, step=step)

    def log_metrics_batch(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        for (mid, step), metric in batch.items():
            self.log_metric(mid=mid, metric=metric, step=step)

    def log_figure(
        self, fid: FigureID, figure: Figure, step: FigureStep = None
    ) -> None:
        """Log figure with given id."""

    def log_figures(self, figures: Figures, step: FigureStep = None) -> None:
        """Log multiple figures at once, by default iterate."""
        for fid, figure in figures.items():
            self.log_figure(fid=fid, figure=figure, step=step)

    def log_figure_batch(self, batch: FiguresBatch) -> None:
        """Log multiple figures samples at arbitrary timesteps at once, by default iterate."""
        for (fid, step), figure in batch.items():
            self.log_figure(fid=fid, figure=figure, step=step)

    def close(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None | bool:
        """Call at the end of the process."""
