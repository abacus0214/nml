"""Useful types for metrics."""

from collections import UserDict

__all__ = [
    "MetricID",
    "ScalarMetric",
    "Metric",
    "Metrics",
    "MetricsBatch",
    "MetricStep",
    "MetricSample",
]

type MetricID = str
type MetricStep = None | int
type MetricSample = tuple[MetricID, MetricStep]

type ScalarMetric = float

type Metric = ScalarMetric

type Metrics = dict[MetricID, Metric]


class MetricsBatch(UserDict[MetricSample, Metric]):
    """Dictionary containing multiple metric values at several steps."""

    @staticmethod
    def from_metrics(metrics: Metrics, step: MetricStep = None) -> "MetricsBatch":
        """Generate a metrics batch from a dict of metrics by repeating a given step multiple timesint."""
        return MetricsBatch({(mid, step): metric for mid, metric in metrics.items()})
