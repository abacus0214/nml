"""MLFlow metric callback."""

from itertools import groupby

from pydantic import Field

from bkd.base.callbacks.abc import EventCallback
from bkd.base.utils.mlflow.client import ExperimentClient, RunClient
from bkd.base.utils.typing.base.pydantic import StandardBaseModel
from bkd.base.utils.typing.eval.metrics import (
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)


class MLFlowMetricCallback(StandardBaseModel, EventCallback):
    """Logs metrics to mlflow."""

    # Parameters
    client: ExperimentClient

    # Settings
    ephimeral_run: bool = Field(default=False)
    run_name: None | str = Field(default=None)

    # State
    current_run: None | RunClient = Field(init=False, default=None)

    def start(self, num_epochs: None | int = None) -> None:
        """Call at the start of process."""
        # Ephimeral run setting means that we have
        # to reload the run no matter what
        if self.ephimeral_run or self.current_run is None:
            self.current_run = self.client.create_run_client(run_name=self.run_name)

    def log_metric(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""
        if self.current_run is not None:
            self.current_run.log_metric(key=mid, value=float(metric), step=step)

    def log_metrics(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        if self.current_run is not None:
            self.current_run.log_metrics(
                {key: float(value) for key, value in metrics.items()},
                step=step,
            )

    def log_metrics_batch(self, batch: MetricsBatch) -> None:
        """Log multiple metrics at once, by default iterate."""
        # Group samples by step
        if self.current_run is not None:
            for step, batch_sub in groupby(batch.items(), lambda it: it[0][1]):
                self.log_metrics(
                    {mid: metric for (mid, _), metric in batch_sub},
                    step=step,
                )

    def close(self) -> None:
        """Call at the end of the process."""
