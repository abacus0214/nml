"""Basic model interpreter interface."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from nml.data.batches.container.base import Batch, BatchABC
from nml.data.batches.interpreter.base import BatchInterpreter
from nml.utils.typing.events import BatchID

__all__ = ["DataLoaderInterpreter"]


class DataLoaderInterpreter[DataseT, BatchT, IpT = Any, TgT = None](ABC):
    """Base data loader container."""

    batch_interpreter: BatchInterpreter[BatchT, IpT, TgT]

    @abstractmethod
    def raw_epoch_iterator(self, dataset: DataseT) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""

    def epoch_iterator(
        self, dataset: DataseT
    ) -> Iterator[tuple[BatchID, BatchABC[BatchT, IpT, TgT]]]:
        """Must be iterable with BatchT types, each iteration is considered an epoch."""
        return (
            (bid, Batch(batch=batch, interpreter=self.batch_interpreter))
            for bid, batch in enumerate(self.raw_epoch_iterator(dataset))
        )

    @abstractmethod
    def len(self, dataset: DataseT) -> int:
        """Must be able to provide length."""
