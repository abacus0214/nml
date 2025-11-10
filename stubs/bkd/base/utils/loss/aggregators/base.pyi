import abc
from abc import ABC, abstractmethod
from typing import Iterable

__all__ = ['LossAggregatorABC']

class LossAggregatorABC[LossT](ABC, metaclass=abc.ABCMeta):
    def aggregate_f(self, losses: Iterable[LossT]) -> float: ...
    @abstractmethod
    def aggregate(self, losses: Iterable[LossT]) -> LossT: ...
    def to_float(self, loss: LossT) -> float: ...
