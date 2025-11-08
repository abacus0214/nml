"""Basic class for batch container."""

from bkd.data.batches.interpreter.base import BatchInterpreter

__all__ = ["BatchContainer"]


class BatchContainer[BatchT, IpT, TgT = None]:
    """Base class for batch container."""

    batch: BatchT
    interpreter: BatchInterpreter[BatchT, IpT, TgT]

    @property
    def ipt(self) -> IpT:
        """Extract model input from batch."""
        return self.interpreter.get_ipt(self.batch)

    @property
    def tgt(self) -> TgT:
        """Extract sample target from batch."""
        return self.interpreter.get_tgt(self.batch)
