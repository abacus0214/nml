"""Implementation of loss aggregator that computes means."""

from typing import Iterable

from nml.utils.loss.aggregators.base import LossAggregatorABC
from numpy import mean as np_mean


class NPMeanAggregator[LossT](LossAggregatorABC[LossT]):
    """Use numpy to compute the mean of the given losses."""

    def aggregate(self, losses: Iterable[LossT]) -> LossT:
        """Compute the mean of the losses."""
        return np_mean(losses)  # type: ignore
