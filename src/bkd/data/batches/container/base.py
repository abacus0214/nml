"""Basic class for batch container."""

from bkd.data.batches.interpreter.base import BatchInterpreter

__all__ = ["BatchContainerABC", "BatchContainer"]


class BatchContainerABC[BatchT, IpT, TgT = None]:
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


class BatchContainer[BatchT, IpT, TgT = None](BatchContainerABC[BatchT, IpT, TgT]):
    """Standard batch container that encapsulates a batch and its reader."""

    def __init__(
        self, batch: BatchT, interpreter: BatchInterpreter[BatchT, IpT, TgT]
    ) -> None:
        """By default simply stores a batch and an interpreter."""
        self.batch = batch
        self.interpreter = interpreter
