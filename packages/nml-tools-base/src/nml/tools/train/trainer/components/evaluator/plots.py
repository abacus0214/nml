"""Evaluator that produces figures."""

from typing import Any

import numpy as np
from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.evaluator.plots import PlotsEvaluator
from plotly import graph_objects as go

__all__ = ["PlotsEvaluator"]


def regression_plot[IpT, TgT](
    fig: go.Figure,
    model: ModelABC[IpT, Any, TgT, Any],
    batch: BatchABC[Any, IpT, TgT],
) -> go.Figure:
    """Add regression points from batch to the figure."""
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
    fig.add_trace(go.Scatter(x=pred, y=tgt))

    return fig
