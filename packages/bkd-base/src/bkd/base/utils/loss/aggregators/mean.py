"""Implementation of loss aggregator that computes means."""

from typing import Iterable

from bkd.base.utils.loss.aggregators.base import LossAggregatorABC
from numpy import mean as np_mean
from torch import Tensor
from torch import mean as torch_mean


class NPMeanAggregator[LossT](LossAggregatorABC[LossT]):
    """Use numpy to compute the mean of the given losses."""

    def aggregate(self, losses: Iterable[LossT]) -> LossT:
        """Compute the mean of the losses."""
        return np_mean(losses)  # type: ignore


class TorchMeanAggregator[LossT: Tensor](LossAggregatorABC[LossT]):
    """Use torch to compute the mean of the given losses."""

    def aggregate(self, losses: Iterable[LossT]) -> LossT:
        """Compute the mean of the losses."""
        return torch_mean(Tensor(losses))  # type: ignore
