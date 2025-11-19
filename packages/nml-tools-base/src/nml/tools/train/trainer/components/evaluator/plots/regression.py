"""Evaluator that produces figures."""

from typing import Any

import numpy as np
from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.tools.train.trainer.components.evaluator.plots.base import PlotGengerator
from nml.train.trainer.components.evaluator.plots import PlotsEvaluator
from plotly import graph_objects as go

__all__ = ["PlotsEvaluator"]


class RegressionPlotGenerator(PlotGengerator):
    """Generator for regression plot."""

    fig: None | go.Figure = None

    def init_fig(self) -> None:
        """Initialize regression plot."""
        # Create figure
        self.fig = go.Figure()

        # Add identity line
        self.fig.update_layout(
            shapes=[
                {
                    "type": "line",
                    "yref": "paper",
                    "xref": "paper",
                    "y0": 0,
                    "y1": 1,
                    "x0": 0,
                    "x1": 1,
                }
            ]
        )

    def update[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> None:
        """Add regression points from batch to the figure."""
        if self.fig is None:
            raise RuntimeError("Need to initialize figure before updating")
        # Get the model predictions and targets
        try:
            # Try to convert to array
            pred = np.squeeze(np.array(model.forward(batch.ipt)))
            tgt = np.squeeze(np.array(batch.tgt))
            # Check that they are 1D
            assert len(pred.shape) == 1, "Cannot support multidimentional predictions."
            assert len(tgt.shape) == 1, "Cannot support multidimentional targets."
        except Exception as exc:
            raise TypeError(
                "Could not convert pred or target to scalar numpy array"
            ) from exc

        # Add points to figure
        self.fig.add_trace(go.Scatter(x=pred, y=tgt))

    def complete(self) -> go.Figure:
        """Return final figure."""
        return self.fig
