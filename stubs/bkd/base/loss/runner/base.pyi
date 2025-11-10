import abc
from abc import ABC, abstractmethod
from bkd.base.data.batches.container.base import BatchABC
from bkd.base.models.container.base import ModelABC
from typing import Any

__all__ = ['LossRunnerABC']

class LossRunnerABC(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def compute[IpT, TgT, LossT](self, batch: BatchABC[Any, IpT, TgT], model: ModelABC[IpT, Any, TgT, LossT]) -> LossT: ...
