"""General interface for plots evaluator producing plotly plots."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Iterable

from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.evaluator.plots import PlotsEvaluator
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import FigureID, Figures
from nml.utils.typing.events import BatchID, EpochID
from plotly import graph_objects as go
from pydantic import Field

__all__ = ["PlotGengerator", "PlotlyEvaluator"]


class PlotGengerator(ABC):
    """Torch eval stile generator for plots."""

    @abstractmethod
    def init_fig(self) -> None:
        """Initialize figure to be updated."""

    @abstractmethod
    def update[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> None:
        """Update figure with new data."""

    @abstractmethod
    def complete(self) -> go.Figure:
        """Return final figure."""

    def reset(self) -> None:
        """Reset figure. By default rerun init."""
        return self.init_fig()


class PlotlyEvaluator(RestrictedBaseModel, PlotsEvaluator):
    """Base class that generate plots using the PlotGenerator interface."""

    plots: dict[FigureID, PlotGengerator]

    # Dictionary determining frequencies of metrics
    frequency: dict[FigureID, int] = Field(
        default_factory=lambda: defaultdict(lambda: 1)
    )

    def evaluate_batch[IpT, TgT](
        self,
        eid: EpochID,
        bid: BatchID,
        model: ModelABC[IpT, Any, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> Figures:
        """Compute results ona given batch (could be training or validation)."""
        # Evaluate on each metric
        for mid, plot in self.plots.items():
            if (eid % self.frequency[mid]) == 0:
                plot.update(
                    model=model,
                    batch=batch,
                )
        return Figures()

    def aggregate_f(
        self,
        results: Iterable[Figures],
    ) -> Figures:
        """Aggregate results from multiple batches."""
        return Figures({str(mid): plot.complete() for mid, plot in self.plots.items()})
