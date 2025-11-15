"""Base class for data container."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from nml.data.batches.container.base import Batch, BatchABC
from nml.data.batches.interpreter.base import BatchInterpreter

__all__ = ["DataLoaderABC"]


class DataLoaderABC[BatchT, IpT = Any, TgT = None](ABC):
    """Base data loader container."""

    name: str
    batch_interpreter: BatchInterpreter[BatchT, IpT, TgT]

    @property
    @abstractmethod
    def raw_epoch_iterator(self) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""

    @property
    def epoch_iterator(self) -> Iterator[BatchABC[BatchT, IpT, TgT]]:
        """Must be iterable with BatchT types, each iteration is considered an epoch."""
        return (
            Batch(batch=batch, interpreter=self.batch_interpreter)
            for batch in self.raw_epoch_iterator
        )
