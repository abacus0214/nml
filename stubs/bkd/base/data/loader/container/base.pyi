import abc
from abc import ABC, abstractmethod
from bkd.base.data.batches.container.base import BatchABC
from bkd.base.data.batches.interpreter.base import BatchInterpreter
from typing import Any, Iterator

__all__ = ['DataLoaderABC']

class DataLoaderABC[BatchT, IpT = Any, TgT = None](ABC, metaclass=abc.ABCMeta):
    name: str
    batch_interpreter: BatchInterpreter[BatchT, IpT, TgT]
    @property
    @abstractmethod
    def epoch_iterator(self) -> Iterator[BatchT]: ...
    def __iter__(self) -> Iterator[BatchABC[BatchT, IpT, TgT]]: ...
