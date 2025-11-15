"""Base class for running evaluation on a split."""

from typing import Any

from nml.data.loader.container.base import DataLoaderABC
from nml.models.container.base import ModelABC
from nml.utils.typing.eval.metrics import Metrics

__all__ = ["SplitEvaluator"]


class SplitEvaluator:
    """Base class for running evaluation on a split."""

    def evaluate_split[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Any],
        dataset: DataLoaderABC[Any, IpT, TgT],
    ) -> Metrics:
        """Compute metrics ona given set (could be training or validation)."""
        return {}
