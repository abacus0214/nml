"""Base torch trainer class."""

from typing import Any

from bkd.itf.torch.models.container.base import TorchModel
from bkd.itf.torch.models.loss.runner.base import TorchLossRunner
from bkd.train.trainer.base import Trainer

__all__ = ["TorchTrainer"]


class TorchTrainer[ModelT: TorchModel[Any, Any, Any], BatchT](
    Trainer[ModelT, BatchT, TorchLossRunner]
):
    """Base torch trainer class."""
