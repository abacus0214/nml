"""Base class for loss aggregator."""

from abc import ABC, abstractmethod
from typing import Iterable

__all__ = ["LossAggregatorABC"]


class LossAggregatorABC[LossT](ABC):
    """Aggregate multiple losses."""

    @abstractmethod
    def __call__(self, losses: Iterable[LossT]) -> LossT:
        """Aggregate a series of losses given."""
