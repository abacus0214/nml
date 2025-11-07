"""MLFlow metric callback."""

from pydantic import Field

from bkd.eval.metrics.callback.abc import MetricCallback
from bkd.utils.mlflow.client import ExperimentClient, RunClient
from bkd.utils.typing.base.pydantic import StandardBaseModel
from bkd.utils.typing.eval.metrics import Metric, MetricID, Metrics


class MLFlowMetricCallback(StandardBaseModel, MetricCallback):
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

    def log_metric(self, mid: MetricID, metric: Metric) -> None:
        """Log metric with given id."""
        if self.current_run is not None:
            self.current_run.log_metric(key=mid, value=float(metric))

    def log_metrics(self, metrics: Metrics) -> None:
        """Log multiple metrics at once, by default iterate."""
        if self.current_run is not None:
            self.current_run.log_metrics(
                {key: float(value) for key, value in metrics.items()}
            )

    def close(self) -> None:
        """Call at the end of the process."""
