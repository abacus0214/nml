"""Evaluator that produces figures."""

from typing import Any

import numpy as np
from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.tools.train.trainer.components.evaluator.plots.base import PlotGengerator
from plotly import graph_objects as go

__all__ = ["DatasetPlotGenerator"]


class DatasetPlotGenerator(PlotGengerator):
    """Generator for 1D dataset plot."""

    fig: None | go.Figure = None

    def init_fig(self) -> None:
        """Initialize regression plot."""
        # Create figure
        self.fig = go.Figure()

    def update[IpT, OutT, TgT](
        self,
        pred: OutT,
        model: ModelABC[IpT, OutT, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> go.Figure:
        """Add regression points from batch to the figure."""
        if self.fig is None:
            self.init_fig()
        if self.fig is None:
            raise RuntimeError("Could not initialize figure.")
        # Get the model predictions and targets
        try:
            # Try to convert to array
            ipt = np.squeeze(np.array(batch.ipt))
            tgt = np.squeeze(np.array(batch.tgt))
            # Check that they are 1D
            assert len(ipt.shape) == 1, (
                f"Cannot support multidimentional predictions of shape {ipt.shape}"
            )
            assert len(tgt.shape) == 1, (
                f"Cannot support multidimentional targets of shape {tgt.shape}."
            )
        except Exception as exc:
            raise TypeError(
                "Could not convert pred or target to scalar numpy array"
            ) from exc

        # Add points to figure
        self.fig.add_trace(go.Scatter(x=ipt, y=tgt))
        return self.fig

    def compute(self) -> go.Figure:
        """Return final figure."""
        if self.fig is None:
            raise RuntimeError("No figure was generated.")
        return self.fig
