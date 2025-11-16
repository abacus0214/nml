"""Test mlflow callback."""

from collections import defaultdict

from hypothesis import given, settings
from nml.callbacks.buffers.metrics_buffer import MetricsBufferCallback
from nml.utils.typing.eval.metrics import MetricsBatch

from .strategies import (
    ExampleMetrics,
    example_metrics,
)


class TestMetricsBufferCallback:
    """Tests for metrics buffer callback."""

    @given(metrics=example_metrics())
    @settings(max_examples=20)
    def test_log_metric(
        self,
        metrics: ExampleMetrics,
    ) -> None:
        """Check that when metrics are logged, nothing changed."""
        # Create callback
        callback = MetricsBufferCallback()

        # Log some metrics
        callback.start()
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                callback.log_metric(mid=metric_key, metric=metric_val, step=step)
        callback.close()

        # Check that they match what mlflow sees
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                assert callback.buffer[(metric_key, step)] == metric_val

    @given(metrics=example_metrics())
    @settings(max_examples=20)
    def test_log_metrics(
        self,
        metrics: ExampleMetrics,
    ) -> None:
        """Check that when metrics are logged, nothing changed."""
        # Create callback
        callback = MetricsBufferCallback()

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

        # Check that they match what mlflow sees
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                assert callback.buffer[(metric_key, step)] == metric_val

    @given(metrics=example_metrics())
    @settings(max_examples=20)
    def test_log_metrics_batch(
        self,
        metrics: ExampleMetrics,
    ) -> None:
        """Check that when metrics are logged, nothing changed."""
        # Create callback
        callback = MetricsBufferCallback()

        # Convert metrics to batch
        metrics_batch: MetricsBatch = MetricsBatch()
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                metrics_batch[(metric_key, step)] = metric_val

        # Log some metrics
        callback.start()
        callback.log_metrics_batch(batch=metrics_batch)
        callback.close()

        # Check that they match what mlflow sees
        for metric_key, metric_vals in metrics.items():
            for step, metric_val in metric_vals.items():
                assert callback.buffer[(metric_key, step)] == metric_val
