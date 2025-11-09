"""Base class for loss aggregator."""

from abc import ABC, abstractmethod
from typing import Iterable, SupportsFloat

__all__ = ["LossAggregatorABC"]


class LossAggregatorABC[LossT](ABC):
    """Aggregate multiple losses."""

    def aggregate_f(self, losses: Iterable[LossT]) -> float:
        """Aggregate a series of losses given. Then cast to float."""
        return self.to_float(self.aggregate(losses=losses))

    @abstractmethod
    def aggregate(self, losses: Iterable[LossT]) -> LossT:
        """Aggregate a series of losses given."""

    def to_float(self, loss: LossT) -> float:
        """Convert the likelihood to a float."""
        if isinstance(loss, SupportsFloat):
            return float(loss)

        raise RuntimeError(f"Could not cast {loss} to float with default strategy.")
