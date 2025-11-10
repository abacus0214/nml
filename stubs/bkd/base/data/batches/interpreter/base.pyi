import abc
from abc import ABC, abstractmethod

__all__ = ['BatchInterpreter']

class BatchInterpreter[BatchT, IpT, TgT = None](ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def get_ipt(self, batch: BatchT) -> IpT: ...
    def get_tgt(self, batch: BatchT) -> TgT: ...
