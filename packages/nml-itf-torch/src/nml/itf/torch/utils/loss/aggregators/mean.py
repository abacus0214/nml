"""Torch loss aggregator."""

from typing import Iterable

from nml.utils.loss.aggregators.base import LossAggregatorABC
from torch import Tensor

__all__ = ["TorchMeanAggregator"]


class TorchMeanAggregator(LossAggregatorABC[Tensor]):
    """Use torch to compute the mean of the given losses."""

    def aggregate(self, losses: Iterable[Tensor]) -> Tensor:
        """Compute the mean of the losses."""
        return torch_mean(Tensor(losses))  # type: ignore
