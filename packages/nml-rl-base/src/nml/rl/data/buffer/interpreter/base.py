"""Extension of data interpreter to represent replay buffer."""

from abc import abstractmethod
from typing import Any, Iterator

from nml.data.loader.interpreter.base import LoaderInterpreter
from nml.rl.utils.types.samples.timestep import TimestepSampleBatch

__all__ = ["BufferInterpreter"]


class BufferInterpreter[DataseT, BatchT, IpT = Any, TgT = None](
    LoaderInterpreter[DataseT, BatchT, IpT, TgT]
):
    """Base data loader container."""

    def raw_epoch_iterator(self, dataset: DataseT) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""
        return (sample.batch for sample in self.raw_epoch_sample_iterator(dataset))

    @abstractmethod
    def raw_epoch_sample_iterator(
        self, dataset: DataseT
    ) -> Iterator[TimestepSampleBatch[BatchT]]:
        """Return iterator to go through batches for a single epoch."""

    @abstractmethod
    def len(self, dataset: DataseT) -> int:
        """Must be able to provide length."""
