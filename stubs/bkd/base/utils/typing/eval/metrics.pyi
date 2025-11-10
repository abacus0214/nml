import numpy as np
from collections import UserDict
from typing import Any

__all__ = ['MetricID', 'ScalarMetric', 'VectorMetric', 'Metric', 'Metrics', 'MetricsBatch', 'MetricStep', 'MetricSample']

type MetricID = str
type MetricStep = None | int
type MetricSample = tuple[MetricID, MetricStep]
type ScalarMetric = float
type VectorMetric = np.ndarray[Any, np.dtype[np.float32]]
type Metric = ScalarMetric | VectorMetric
type Metrics = dict[MetricID, Metric]
class MetricsBatch(UserDict[MetricSample, Metric]):
    @staticmethod
    def from_metrics(metrics: Metrics, step: MetricStep = None) -> MetricsBatch: ...
