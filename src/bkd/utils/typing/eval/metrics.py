"""Useful types for metrics."""

from typing import Any, TypeAlias

import numpy as np

__all__ = ["MetricID", "ScalarMetric", "VectorMetric", "Metric", "Metrics"]

MetricID: TypeAlias = str

ScalarMetric: TypeAlias = float
VectorMetric: TypeAlias = np.ndarray[Any, np.dtype[np.float32]]

Metric: TypeAlias = ScalarMetric | VectorMetric

Metrics: TypeAlias = dict[MetricID, Metric]
