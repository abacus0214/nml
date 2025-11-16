"""Base class for data container."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from nml.data.batches.container.base import Batch, BatchABC
from nml.data.batches.interpreter.base import BatchInterpreter
from nml.utils.typing.events import BatchID

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
    def epoch_iterator(self) -> Iterator[tuple[BatchID, BatchABC[BatchT, IpT, TgT]]]:
        """Must be iterable with BatchT types, each iteration is considered an epoch."""
        return (
            (bid, Batch(batch=batch, interpreter=self.batch_interpreter))
            for bid, batch in enumerate(self.raw_epoch_iterator)
        )

    @abstractmethod
    def __len__(self) -> int:
        """Must be able to provide length."""
