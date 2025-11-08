"""Base class for data container."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from bkd.data.batches.container.base import Batch, BatchABC
from bkd.data.batches.interpreter.base import BatchInterpreter

__all__ = ["DataLoaderABC"]


class DataLoaderABC[BatchT, IpT = Any, TgT = None](ABC):
    """Base data loader container."""

    @property
    @abstractmethod
    def batch_interpreter(self) -> BatchInterpreter[BatchT, IpT, TgT]:
        """Get the batch reader for the batch type contained in this dataset."""

    @property
    @abstractmethod
    def epoch_iterator(self) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""

    def __iter__(self) -> Iterator[BatchABC[BatchT, IpT, TgT]]:
        """Must be iterable with BatchT types, each iteration is considered an epoch."""
        return map(
            lambda batch: Batch(batch=batch, interpreter=self.batch_interpreter),
            self.epoch_iterator,
        )
