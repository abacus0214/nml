"""Test mlflow callback."""

from pathlib import Path

from _pytest.tmpdir import TempPathFactory
from mlflow.client import MlflowClient
from nml.callbacks.logging.mlflow import MLFlowMetricCallback
from nml.utils.mlflow.client import ExperimentClient
from pytest import fixture

TEST_EXPERIMENT_NAME = "test_experiment"
TEST_RUN_NAME = "test_run"


@fixture(scope="class")
def experiment_name() -> str:
    """Experiment name to be used in client."""
    return TEST_EXPERIMENT_NAME


@fixture(scope="class")
def mlflow_path(tmp_path_factory: TempPathFactory) -> Path:
    """Generate temporary mlflow path."""
    return tmp_path_factory.mktemp("mlflow")


@fixture(scope="class")
def experiment_client(mlflow_path: Path) -> ExperimentClient:
    """Create client."""
    return ExperimentClient(
        in_experiment_name=TEST_EXPERIMENT_NAME,
        client=MlflowClient(tracking_uri=mlflow_path.as_uri()),
    )


class TestMlflowCallback:
    """Tests for mlflow callback."""

    def test_client(
        self, experiment_client: ExperimentClient, experiment_name: str
    ) -> None:
        """Check client consistency."""
        # Load experiment
        experiment_client.experiment_id

        # Check that experiment id is what is expetected given the name
        exp_from_name = experiment_client.client.get_experiment_by_name(experiment_name)
        assert exp_from_name is not None
        assert experiment_client.experiment_id == exp_from_name.experiment_id

        # Check that experiment id was not edited
        experiment_client_cp = ExperimentClient(
            client=experiment_client.client,
            in_experiment_id=experiment_client.experiment_id,
        )
        assert experiment_client_cp.experiment_id == experiment_client.experiment_id

    def test_mlflow_client_result(
        self,
        experiment_client: ExperimentClient,
    ) -> None:
        """Check that mlflow callback via Mlflow.

        Check that logging using the mlflow metric callback results in properly logged metric on mlflow.
        """
        # Create callback
        callback = MLFlowMetricCallback(
            run_name=TEST_RUN_NAME, ephimeral_run=False, client=experiment_client
        )

        # Log some metrics
