"""Base torch trainer class."""

from typing import Any

from torch.utils.data import DataLoader

from bkd.itf.torch.models.container.base import TorchModel
from bkd.train.trainer.base import Trainer

__all__ = ["TorchTrainer"]


class TorchTrainer[ModelT: TorchModel[Any, Any, Any], BatchT](
    Trainer[ModelT, DataLoader[BatchT]]
):
    """Base torch trainer class."""
