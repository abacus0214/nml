"""Base class for running evaluation on a split."""

from collections import defaultdict
from typing import Any, Iterable

from nml.data.batches.container.base import BatchABC
from nml.models.container.base import ModelABC
from nml.models.interpreter.base import PredictionType
from nml.tools.utils.frequencer.base import FrequencerABC
from nml.tools.utils.frequencer.standard import Every
from nml.train.trainer.components.evaluator.metrics import MetricsEvaluator
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import Metric, MetricID, Metrics
from nml.utils.typing.events import BatchID, EpochID
from pydantic import Field
from torcheval.metrics import Metric as TorchEvalMetric

__all__ = ["TorchEvalEvaluator"]


class TorchEvalEvaluator(RestrictedBaseModel, MetricsEvaluator):
    """Base class for running evaluation on a split."""

    metrics: dict[MetricID, TorchEvalMetric[Metric]] = Field(default_factory=dict)

    # Dictionary that can be used to customize which metric uses which method
    # to compute its input from the model output
    pred_type: dict[MetricID, PredictionType] = Field(
        default_factory=lambda: defaultdict(lambda: PredictionType.OUTPUT)
    )

    # Dictionary determining frequencies of metrics
    frequency: dict[MetricID, FrequencerABC] = Field(default_factory=dict)
    default_frequencer: FrequencerABC = Field(default_factory=lambda: Every(freq=1))

    def evaluate_batch[IpT, OutT, TgT](
        self,
        eid_max: EpochID,
        eid: EpochID,
        bid: BatchID,
        pred: OutT,
        model: ModelABC[IpT, OutT, TgT, Any],
        batch: BatchABC[Any, IpT, TgT],
    ) -> Metrics:
        """Compute metrics ona given batch (could be training or validation)."""
        # Initialize dict of computed metrics
        metrics = Metrics()
        # Evaluate on each metric
        for mid, metric in self.metrics.items():
            if self.frequency.get(mid, self.default_frequencer)(
                eid=eid, eid_max=eid_max
            ):
                # TODO: add possibility to customize target
                metric.update(
                    model.interpreter(model_out=pred, tp=self.pred_type[mid]),
                    batch.tgt,
                )
                metrics[mid] = True
        return metrics

    @staticmethod
    def extract_metric(metric: TorchEvalMetric[Metric]) -> Metric:
        """Compute and reset metric."""
        ret = metric.compute()
        metric.reset()
        return ret

    def aggregate_f(
        self,
        results: Iterable[Metrics],
    ) -> Metrics:
        """Aggregate metrics from multiple batches."""
        # Get all keys that have been computed
        # assume that it is the same for every baatch
        available_mids = next(iter(results)).keys()
        # Generate metrics
        return Metrics(
            {
                str(mid): self.extract_metric(metric)
                for mid, metric in self.metrics.items()
                if mid in available_mids
            }
        )
