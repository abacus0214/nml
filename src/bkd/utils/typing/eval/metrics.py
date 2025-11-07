"""Useful types for metrics."""

from collections import UserDict
from typing import Any, TypeAlias

import numpy as np

__all__ = [
    "MetricID",
    "ScalarMetric",
    "VectorMetric",
    "Metric",
    "Metrics",
    "MetricsBatch",
    "MetricStep",
    "MetricSample",
]

MetricID: TypeAlias = str
MetricStep: TypeAlias = None | int
MetricSample: TypeAlias = tuple[MetricID, MetricStep]

ScalarMetric: TypeAlias = float
VectorMetric: TypeAlias = np.ndarray[Any, np.dtype[np.float32]]

Metric: TypeAlias = ScalarMetric | VectorMetric

Metrics: TypeAlias = dict[MetricID, Metric]


class MetricsBatch(UserDict[MetricSample, Metric]):
    """Dictionary containing multiple metric values at several steps."""

    @staticmethod
    def from_metrics(metrics: Metrics, step: MetricStep = None) -> "MetricsBatch":
        """Generate a metrics batch from a dict of metrics by repeating a given step multiple timesint."""
        return MetricsBatch({(mid, step): metric for mid, metric in metrics.items()})
