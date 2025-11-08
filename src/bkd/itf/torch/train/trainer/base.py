"""Base torch trainer class."""


from torch import Tensor

from bkd.train.trainer.base import Trainer

__all__ = ["TorchTrainer"]


class TorchTrainer(Trainer[Tensor]):
    """Base torch trainer class."""
