"""Useful types for metrics."""

from collections import UserDict

from matplotlib.figure import Figure as MPLFig
from plotly.graph_objects import Figure as PLFig

__all__ = [
    "MetricID",
    "ScalarMetric",
    "Metric",
    "Metrics",
    "MetricsBatch",
    "MetricStep",
    "MetricSample",
    "FigureID",
    "Figure",
    "Figures",
]

type MetricID = str
type MetricStep = None | int
type MetricSample = tuple[MetricID, MetricStep]

type ScalarMetric = float

type Metric = ScalarMetric


class Metrics(UserDict[MetricID, Metric]):
    """Dictionary containing set of metrics (referring to the same timestep)."""

    def add_prefix(self, prefix: str) -> "Metrics":
        """Add prefix to each metric name."""
        return Metrics({f"{prefix}/{mid}": metric for mid, metric in self.items()})


class MetricsBatch(UserDict[MetricSample, Metric]):
    """Dictionary containing multiple metric values at several steps."""

    @staticmethod
    def from_metrics(metrics: Metrics, step: MetricStep = None) -> "MetricsBatch":
        """Generate a metrics batch from a dict of metrics by repeating a given step multiple timesint."""
        return MetricsBatch({(mid, step): metric for mid, metric in metrics.items()})


type FigureID = str
type Figure = MPLFig | PLFig


class Figures(UserDict[FigureID, Figure]):
    """Dictionary of figures."""
