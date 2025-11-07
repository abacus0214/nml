"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod
from typing import Any, Iterable

from bkd.eval.metrics.callback.abc import MetricCallback

__all__ = ["Trainer"]


class Trainer[ModelT, DatasetT: Iterable[Any]](ABC):
    """Base class that defines the interface for a trainer."""

    @abstractmethod
    def train(
        self,
        model: ModelT,
        dataset: DatasetT,
        callback: MetricCallback = MetricCallback(),
    ) -> None:
        """Perform training loop."""
