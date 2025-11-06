"""Interface for evaluation callback."""

from abc import ABC, abstractmethod

from bkd.utils.typing.eval.metrics import Metric, MetricID, Metrics

__all__ = ["MetricCallback"]


class MetricCallback(ABC):
    """Callback to be executed when metric(s) is/are computed."""

    @abstractmethod
    def log_metric(self, mid: MetricID, metric: Metric) -> None:
        """Log metric with given id."""

    def log_metrics(self, metrics: Metrics) -> None:
        """Log multiple metrics at once, by default iterate."""
        for mid, metric in metrics.items():
            self.log_metric(mid=mid, metric=metric)
