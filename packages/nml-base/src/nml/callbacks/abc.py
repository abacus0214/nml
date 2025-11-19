"""Interface for evaluation callback."""

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

    # TODO: turn into context manager to be able to catch exception (e.g. set mlflow run as failed)

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

    def close(self) -> None:
        """Call at the end of the process."""
