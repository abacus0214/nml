"""Base class for data container."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from bkd.data.batches.interpreter.base import BatchInterpreter

__all__ = ["DataLoader"]


class DataLoader[BatchT, IpT = Any, TgT = None](ABC):
    """Base data loader container."""

    @property
    @abstractmethod
    def batch_interpreter(self) -> BatchInterpreter[BatchT, IpT, TgT]:
        """Get the batch reader for the batch type contained in this dataset."""

    @property
    @abstractmethod
    def epoch_iterator(self) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""

    def __iter__(self) -> Iterator[BatchT]:
        """Must be iterable with BatchT types, each iteration is considered an epoch."""
        return self.epoch_iterator
