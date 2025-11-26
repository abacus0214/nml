"""Standard interpreter for batch tuples."""

from nml.data.batches.interpreter.base import BatchInterpreter

__all__ = ["TupleBatchInterpreter"]


class TupleBatchInterpreter[IpT, TgT = None](
    BatchInterpreter[tuple[IpT, TgT], IpT, TgT]
):
    """Base class for batch interpreter."""

    def get_ipt(self, batch: tuple[IpT, TgT]) -> IpT:
        """Extract the model input from the batch."""
        return batch[0]

    def get_tgt(self, batch: tuple[IpT, TgT]) -> TgT:
        """Extract the sample target from the batch."""
        return batch[1]
