"""General interface for plots evaluator producing plotly plots."""

from abc import ABC, abstractmethod
from typing import Any, Iterable

from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.tools.utils.frequencer.base import FrequencerABC
from nml.tools.utils.frequencer.standard import Every
from nml.train.trainer.components.evaluator.plots import PlotsEvaluator
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import FigureID, Figures
from nml.utils.typing.events import TrainingStepID
from plotly import graph_objects as go
from pydantic import Field

__all__ = ["PlotGengerator", "PlotlyEvaluator"]


class PlotGengerator(ABC):
    """Torch eval stile generator for plots."""

    @abstractmethod
    def init_fig(self) -> None:
        """Initialize figure to be updated."""

    @abstractmethod
    def update[IpT, OutT, TgT](
        self,
        pred: OutT,
        model: ModelABC[IpT, OutT, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> go.Figure:
        """Update figure with new data."""

    def complete(self) -> go.Figure:
        """Return final figure."""
        ret = self.compute()
        self.reset()
        return ret

    @abstractmethod
    def compute(self) -> go.Figure:
        """Return final figure."""

    def reset(self) -> None:
        """Reset figure. By default rerun init."""
        return self.init_fig()


class PlotlyEvaluator(RestrictedBaseModel, PlotsEvaluator):
    """Base class that generate plots using the PlotGenerator interface."""

    plots: dict[FigureID, PlotGengerator]

    # Dictionary determining frequencies of metrics
    frequency: dict[FigureID, FrequencerABC] = Field(
        default_factory=dict,
    )
    default_frequencer: FrequencerABC = Field(default_factory=lambda: Every(freq=1))

    def evaluate_batch[IpT, OutT, TgT](
        self,
        step_id: TrainingStepID,
        pred: OutT,
        model: ModelABC[IpT, OutT, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> Figures:
        """Compute results ona given batch (could be training or validation)."""
        # Evaluate on each metric
        return Figures(
            {
                fid: plot.update(pred=pred, model=model, batch=batch)
                for fid, plot in self.plots.items()
                if self.frequency.get(fid, self.default_frequencer)(step_id=step_id)
            }
        )

    def aggregate_f(
        self,
        results: Iterable[Figures],
    ) -> Figures:
        """Aggregate results from multiple batches."""
        # Get available keys
        # Assume they are the same for every batch.
        available_fids = next(iter(results)).keys()
        # Return plots
        return Figures(
            {
                str(fid): plot.complete()
                for fid, plot in self.plots.items()
                if fid in available_fids
            }
        )
