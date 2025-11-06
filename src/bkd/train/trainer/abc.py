"""Main module with bulk of trainer loop."""

from abc import ABC, abstractmethod

from torch.nn import Module
from torch.utils.data import DataLoader

from bkd.eval.metrics.callback.abc import MetricCallback

__all__ = ["Trainer"]


class Trainer[BatchT](ABC):
    """Base class that defines the interface for a trainer."""

    @abstractmethod
    def train(
        self,
        model: Module,
        dataset: DataLoader[BatchT],
        callback: MetricCallback = MetricCallback(),
    ) -> None:
        """Perform training loop."""
