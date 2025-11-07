"""Standard torch trainer."""

from torch.nn import Module
from torch.utils.data import DataLoader

from bkd.train.trainer.base import Trainer

__all__ = ["TorchTrainer"]


class TorchTrainer[BatchT](Trainer[Module, DataLoader[BatchT]]):
    """Standard torch trainer."""
