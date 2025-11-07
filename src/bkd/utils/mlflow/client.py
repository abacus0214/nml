"""Utilities for mlflow client(s)."""

from mlflow.client import MlflowClient
from mlflow.utils.async_logging.run_operations import RunOperations

from bkd.utils.typing.base.pydantic import RestrictedBaseModel
from bkd.utils.typing.mlflow import (
    MLFlowDatasetDigest,
    MLFlowDatasetName,
    MLFlowExperimentID,
    MLFlowMetricKey,
    MLFlowMetricVal,
    MLFlowModelID,
    MLFlowRunID,
    MLFlowRunName,
    MLFlowStepType,
    MLFlowTagsDict,
    MLFlowTimeType,
    Run,
)

__all__ = ["ExperimentClient", "RunClient"]


class ExperimentClient(RestrictedBaseModel):
    """Wrapper of MLFlow client that fixes the experiment."""

    experiment_id: MLFlowExperimentID
    client: MlflowClient

    def create_run(
        self,
        start_time: None | MLFlowTimeType = None,
        tags: None | MLFlowTagsDict = None,
        run_name: None | MLFlowRunName = None,
    ) -> Run:
        """Forward to client.create_run using given experiment id."""
        return self.client.create_run(
            experiment_id=self.experiment_id,
            start_time=start_time,
            tags=tags,
            run_name=run_name,
        )

    def set_run_id(self, run_id: MLFlowRunID) -> "RunClient":
        """Generate RunClient with given run id."""
        return RunClient(
            run_id=run_id,
            experiment_id=self.experiment_id,
            client=self.client,
        )


class RunClient(ExperimentClient):
    """Wrapper of MLFlow client that fixes the experiment."""

    run_id: MLFlowRunID

    def log_metric(
        self,
        key: MLFlowMetricKey,
        value: MLFlowMetricVal,
        timestamp: None | MLFlowTimeType = None,
        step: None | MLFlowStepType = None,
        synchronous: None | bool = None,
        dataset_name: None | MLFlowDatasetName = None,
        dataset_digest: None | MLFlowDatasetDigest = None,
        model_id: None | MLFlowModelID = None,
    ) -> None | RunOperations:
        """Forward to client log metric."""
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
