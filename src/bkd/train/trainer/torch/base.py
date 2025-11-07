"""Standard torch trainer."""

from torch.nn import Module

from bkd.train.trainer.base import Trainer

__all__ = ["TorchTrainer"]


class TorchTrainer[BatchT](Trainer[Module, BatchT]):
    """Standard torch trainer."""
