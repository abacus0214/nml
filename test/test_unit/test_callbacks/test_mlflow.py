"""Test mlflow callback."""

from collections import defaultdict
from pathlib import Path

from _pytest.tmpdir import TempPathFactory
from hypothesis import given, settings
from mlflow.client import MlflowClient
from nml.callbacks.logging.mlflow import MLFlowMetricCallback
from nml.utils.mlflow.client import ExperimentClient
from nml.utils.typing.eval.metrics import MetricsBatch
from pytest import fixture

from .strategies import (
    ExampleMetricHistory,
    ExampleMetrics,
    example_metrics,
)

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

    @given(metrics=example_metrics())
    @settings(max_examples=20)
    def test_log_metric(
        self,
        metrics: ExampleMetrics,
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
        callback.start()
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                callback.log_metric(mid=metric_key, metric=metric_val, step=step)
        callback.close()

        # Try to retrieve run id
        assert callback.current_run
        run_id = callback.current_run.run_id

        # Check that they match what mlflow sees
        for metric_key, metric_vals in metrics.items():
            # Get metrics from mlflow
            history = self.get_metric_history(experiment_client, run_id, metric_key)
            # Compare
            assert history == metric_vals

    @given(metrics=example_metrics())
    @settings(max_examples=20)
    def test_log_metrics(
        self,
        metrics: ExampleMetrics,
        experiment_client: ExperimentClient,
    ) -> None:
        """Check that mlflow callback via Mlflow.

        Check that logging using the mlflow metric callback results in properly logged metric on mlflow.
        """
        # Create callback
        callback = MLFlowMetricCallback(
            run_name=TEST_RUN_NAME, ephimeral_run=False, client=experiment_client
        )

        # Group metrics by steps
        metrics_per_step: dict[int, dict[str, float]] = defaultdict(dict)
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                metrics_per_step[step][metric_key] = metric_val

        # Log some metrics
        callback.start()
        for step, metrics_vals in metrics_per_step.items():
            callback.log_metrics(metrics=metrics_vals, step=step)
        callback.close()

        # Try to retrieve run id
        assert callback.current_run
        run_id = callback.current_run.run_id

        # Check that they match what mlflow sees
        for metric_key, metric_vals in metrics.items():
            # Get metrics from mlflow
            history = self.get_metric_history(experiment_client, run_id, metric_key)
            # Compare
            assert history == metric_vals

    @given(metrics=example_metrics())
    @settings(max_examples=20)
    def test_log_metrics_batch(
        self,
        metrics: ExampleMetrics,
        experiment_client: ExperimentClient,
    ) -> None:
        """Check that mlflow callback via Mlflow.

        Check that logging using the mlflow metric callback results in properly logged metric on mlflow.
        """
        # Create callback
        callback = MLFlowMetricCallback(
            run_name=TEST_RUN_NAME, ephimeral_run=False, client=experiment_client
        )

        # Convert metrics to batch
        metrics_batch: MetricsBatch = MetricsBatch()
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                metrics_batch[(metric_key, step)] = metric_val

        # Log some metrics
        callback.start()
        callback.log_metrics_batch(batch=metrics_batch)
        callback.close()

        # Try to retrieve run id
        assert callback.current_run
        run_id = callback.current_run.run_id

        # Check that they match what mlflow sees
        for metric_key, metric_vals in metrics.items():
            # Get metrics from mlflow
            history = self.get_metric_history(experiment_client, run_id, metric_key)
            # Compare
            assert history == metric_vals

    @staticmethod
    def get_metric_history(
        experiment_client: ExperimentClient, run_id: str, metric_key: str
    ) -> ExampleMetricHistory:
        """Get the metric history from mlflow and reformat."""
        # Get raw history
        raw_history = experiment_client.client.get_metric_history(run_id, metric_key)

        # Initialize output dict
        history: ExampleMetricHistory = {}
        # Fill in history
        for sample in raw_history:
            history[sample.step] = sample.value

        return history
