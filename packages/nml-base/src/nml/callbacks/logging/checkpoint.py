"""Callback to store model checkpoint."""

from copy import deepcopy
from operator import gt
from typing import Any, Callable

from nml.callbacks.abc import EventCallback
from nml.callbacks.stack import CallbackStack
from nml.models.container.base import ModelABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import (
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from pydantic import Field

__all__ = ["CheckpointCallback"]

type NewMetric = Metric
type OldMetric = Metric


class CheckpointCallback(RestrictedBaseModel, CallbackStack):
    """Store best or latest model.

    This callback will hide models from the docorated calllback
    until closure, where only the latest tracked checkpoint
    will be forwarded.
    """

    callback: EventCallback = Field(default_factory=EventCallback)

    select_by: MetricID
    # Return true if the new metric is better than the old metric value
    # If this is set to `[] -> True` and `allow_backdating=False` then
    # this callback will effectively record the latest checkpoint
    comparator: Callable[[NewMetric, OldMetric], bool] = Field(default=gt)

    # Set to true if you can provide a new checkpoint candidate from the past
    # by default it is set to `False`, therefore checkpoints that come from an
    # older timestep than the current one are ignored
    allow_backdating: bool = False

    checkpoint: None | ModelABC[Any, Any, Any, Any] = Field(init=False, default=None)
    checkpoint_metric: None | Metric = Field(init=False, default=None)
    checkpoint_step: None | MetricStep = Field(init=False, default=None)

    def log_model(
        self, model: ModelABC[Any, Any, Any, Any], step: MetricStep = None
    ) -> None:
        """Log a model."""
        if step == self.checkpoint_step:
            self.checkpoint = deepcopy(model)

    def log_metric_aux(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""
        self.log_metrics_aux(metrics=Metrics({mid: metric}), step=step)

    def log_metrics_aux(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        self.log_metrics_batch_aux(MetricsBatch.from_metrics(metrics, step=step))

    def log_metrics_batch_aux(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        for (mid, step), metric in batch.items():
            # Skip metrics that are not tracked
            # Skip backdated timesteps if backdating is not allowed
            if mid != self.select_by or (
                not self.allow_backdating
                and self.checkpoint_step is not None
                and (step is None or step < self.checkpoint_step)
            ):
                continue

            # If first checkpoint or better checkpoint
            if (self.checkpoint_metric is None) or self.comparator(
                metric, self.checkpoint_metric
            ):
                # Update checkpoint metadata
                self.checkpoint_step = step
                self.checkpoint_metric = metric

    def close_aux(self) -> None:
        """Call at the end of the process."""
        if self.checkpoint is not None:
            self.callback.log_model(self.checkpoint, step=self.checkpoint_step)
