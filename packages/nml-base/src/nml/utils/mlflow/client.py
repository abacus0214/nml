"""Utilities for mlflow client(s)."""

from functools import cached_property
from typing import Any, Self, Sequence

from matplotlib.figure import Figure as MPLFig
from mlflow.client import MlflowClient
from mlflow.entities import Experiment, RunStatus
from mlflow.tracking.fluent import (
    _get_model_ids_for_new_metric_if_exist,
    get_active_model_id,
)
from mlflow.utils.async_logging.run_operations import RunOperations
from mlflow.utils.time import get_current_time_millis
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.mlflow import (
    MLFlowDataset,
    MLFlowMetric,
    MLFlowParam,
    MLFlowRun,
    MLFlowRunTag,
    MLFlowTagsDict,
)
from plotly.graph_objects import Figure as PLFig
from pydantic import model_validator

__all__ = ["ExperimentClient", "RunClient"]


class ExperimentClient(RestrictedBaseModel):
    """Wrapper of MLFlow client that fixes the experiment."""

    in_experiment_name: None | str = None
    in_experiment_id: None | str = None
    client: MlflowClient

    @model_validator(mode="after")
    def check_experiment_given(self) -> Self:
        """Check that either experiment name or id are given."""
        if self.in_experiment_id is None and self.in_experiment_name is None:
            raise ValueError("Must provide either experiment name or id.")

        return self

    @cached_property
    def experiment_id(self) -> str:
        """Retrieve experiment id."""
        if self.in_experiment_id is not None:
            exp: None | Experiment = self.client.get_experiment(self.in_experiment_id)
        elif self.in_experiment_name is not None:
            exp = self.client.get_experiment_by_name(self.in_experiment_name)
        else:
            raise RuntimeError("Failure to retrieve information about experiment.")

        if exp is not None:
            return str(exp.experiment_id)

        if self.in_experiment_name is None:
            raise RuntimeError("No experiment found, could not create without name.")
        return self.client.create_experiment(self.in_experiment_name)

    def create_run(
        self,
        start_time: None | int = None,
        tags: None | MLFlowTagsDict = None,
        run_name: None | str = None,
    ) -> MLFlowRun:
        """Forward to client.create_run using given experiment id."""
        return self.client.create_run(
            experiment_id=self.experiment_id,
            start_time=start_time,
            tags=tags,
            run_name=run_name,
        )

    def create_run_client(
        self,
        start_time: None | int = None,
        tags: None | MLFlowTagsDict = None,
        run_name: None | str = None,
    ) -> "RunClient":
        """Forward to client.create_run using given experiment id."""
        return self.set_run_id(
            run_id=self.create_run(
                start_time=start_time,
                tags=tags,
                run_name=run_name,
            ).info.run_id
        )

    def set_run_id(self, run_id: str) -> "RunClient":
        """Generate RunClient with given run id."""
        return RunClient(
            run_id=run_id,
            in_experiment_name=self.in_experiment_name,
            in_experiment_id=self.in_experiment_id,
            client=self.client,
        )


class RunClient(ExperimentClient):
    """Wrapper of MLFlow client that fixes the experiment."""

    run_id: str

    @property
    def run(self) -> MLFlowRun:
        """Get run from id."""
        return self.client.get_run(self.run_id)

    def set_status(self, run_status: int) -> None:
        """Set the run status."""
        self.client.update_run(self.run_id, RunStatus.to_string(run_status))

    def log_metric(
        self,
        key: str,
        value: float,
        timestamp: None | int = None,
        step: None | int = None,
        synchronous: None | bool = None,
        dataset_name: None | str = None,
        dataset_digest: None | str = None,
        model_id: None | str = None,
    ) -> None | RunOperations:
        """Forward to client log metric."""
        # TODO: sanitize names
        return self.client.log_metric(
            run_id=self.run_id,
            key=key,
            value=value,
            timestamp=timestamp,
            step=step,
            synchronous=synchronous,
            dataset_name=dataset_name,
            dataset_digest=dataset_digest,
            model_id=model_id,
        )

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: None | int = None,
        synchronous: None | bool = None,
        timestamp: None | int = None,
        model_id: None | str = None,
        dataset: None | MLFlowDataset = None,
    ) -> None | RunOperations:
        """Recreate log metrics but for client."""
        # TODO: sanitize names
        # Get timestamp in milliseconds
        time_millis = timestamp or get_current_time_millis()
        # Initalize step to 0
        int_step = step or 0

        # Extract dataset metadata
        dataset_name = dataset.name if dataset is not None else None
        dataset_digest = dataset.digest if dataset is not None else None
        # Extract model id
        extracted_model_id = model_id or get_active_model_id()
        model_ids = (
            [extracted_model_id]
            if extracted_model_id is not None
            else (_get_model_ids_for_new_metric_if_exist(self.run, int_step) or [None])  # type: ignore
        )

        # Log batch of metrics
        return self.log_batch(
            metrics=[
                MLFlowMetric(
                    key=key,
                    value=value,
                    timestamp=time_millis,
                    step=int_step,
                    model_id=run_model_id,
                    dataset_name=dataset_name,
                    dataset_digest=dataset_digest,
                    run_id=self.run_id,
                )
                for key, value in metrics.items()
                for run_model_id in model_ids
            ],
            params=[],
            tags=[],
            synchronous=synchronous,
        )

    def log_batch(
        self,
        metrics: Sequence[MLFlowMetric] = (),
        params: Sequence[MLFlowParam] = (),
        tags: Sequence[MLFlowRunTag] = (),
        synchronous: None | bool = None,
    ) -> None | RunOperations:
        """Log batch of metrics."""
        return self.client.log_batch(
            run_id=self.run_id,
            metrics=metrics,
            params=params,
            tags=tags,
            synchronous=synchronous,
        )

    def log_figure(
        self,
        figure: PLFig | MPLFig,
        artifact_file: str,
        *,
        save_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Forward log figure."""
        self.client.log_figure(
            run_id=self.run_id,
            figure=figure,
            artifact_file=artifact_file,
            save_kwargs=save_kwargs,
        )
