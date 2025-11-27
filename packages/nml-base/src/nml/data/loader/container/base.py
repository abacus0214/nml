"""Base class for data container."""

from abc import ABC, abstractmethod
from typing import Any, Iterator

from nml.data.batches.container.base import BatchABC
from nml.data.batches.interpreter.base import BatchInterpreter
from nml.data.loader.interpreter.base import LoaderInterpreter
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.events import BatchID

__all__ = ["LoaderContainer", "LoaderContainerABC"]


class LoaderContainerABC[DatasetT, BatchT, IpT = Any, TgT = None](ABC):
    """Base data loader container."""

    loader: DatasetT
    loader_interpreter: LoaderInterpreter[DatasetT, BatchT, IpT, TgT]

    @property
    @abstractmethod
    def batch_interpreter(self) -> BatchInterpreter[BatchT, IpT, TgT]:
        """Return batch interpreter for batches in dataset."""

    @property
    @abstractmethod
    def raw_epoch_iterator(self) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""
        return self.loader_interpreter.raw_epoch_iterator(self.loader)

    @property
    @abstractmethod
    def epoch_iterator(self) -> Iterator[tuple[BatchID, BatchABC[BatchT, IpT, TgT]]]:
        """Must be iterable with BatchT types, each iteration is considered an epoch."""
        return self.loader_interpreter.epoch_iterator(self.loader)

    @property
    @abstractmethod
    def len(self) -> int:
        """Must be able to provide length."""

    @abstractmethod
    def __len__(self) -> int:
        """Must be able to provide length."""


class LoaderContainer[DatasetT, BatchT, IpT = Any, TgT = None](
    RestrictedBaseModel, LoaderContainerABC[DatasetT, BatchT, IpT, TgT]
):
    """Base data loader container."""

    loader: DatasetT
    loader_interpreter: LoaderInterpreter[DatasetT, BatchT, IpT, TgT]

    @property
    def batch_interpreter(self) -> BatchInterpreter[BatchT, IpT, TgT]:
        """Return batch interpreter for batches in dataset."""
        return self.loader_interpreter.batch_interpreter

    @property
    def raw_epoch_iterator(self) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""
        return self.loader_interpreter.raw_epoch_iterator(self.loader)

    @property
    def epoch_iterator(self) -> Iterator[tuple[BatchID, BatchABC[BatchT, IpT, TgT]]]:
        """Must be iterable with BatchT types, each iteration is considered an epoch."""
        return self.loader_interpreter.epoch_iterator(self.loader)

    @property
    def len(self) -> int:
        """Must be able to provide length."""
        return len(self)

    def __len__(self) -> int:
        """Must be able to provide length."""
        return self.loader_interpreter.len(self.loader)
