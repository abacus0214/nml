"""Base torch trainer class."""


from bkd.train.trainer.base import Trainer
from torch import Tensor

__all__ = ["TorchTrainer"]


class TorchTrainer(Trainer[Tensor]):
    """Base torch trainer class."""
