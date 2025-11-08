"""Base class for batch interpreter."""

from abc import ABC, abstractmethod

__all__ = ["BatchInterpreter"]


class BatchInterpreter[BatchT, IpT, TgT = None](ABC):
    """Base class for batch interpreter."""

    @abstractmethod
    def get_ipt(self, batch: BatchT) -> IpT:
        """Extract the model input from the batch."""

    def get_tgt(self, batch: BatchT) -> TgT:
        """Extract the sample target from the batch."""
        raise NotImplementedError(f"{self} does not support target extraction")
