from _typeshed import Incomplete
from bkd.base.data.batches.interpreter.base import BatchInterpreter

__all__ = ['BatchABC', 'Batch']

class BatchABC[BatchT, IpT, TgT = None]:
    batch: BatchT
    interpreter: BatchInterpreter[BatchT, IpT, TgT]
    @property
    def ipt(self) -> IpT: ...
    @property
    def tgt(self) -> TgT: ...

class Batch[BatchT, IpT, TgT = None](BatchABC[BatchT, IpT, TgT]):
    batch: Incomplete
    interpreter: Incomplete
    def __init__(self, batch: BatchT, interpreter: BatchInterpreter[BatchT, IpT, TgT]) -> None: ...
