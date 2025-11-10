"""Component that maps a model to its optimizer."""

from abc import ABC, abstractmethod
from typing import Any

from bkd.base.models.container.base import ModelABC
from torch.optim import Optimizer

__all__ = ["OptimizerMapABC"]


class OptimizerMapABC(ABC):
    """Map from model to optimizer."""

    @abstractmethod
    def get_optimizer(self, model: ModelABC[Any, Any, Any, Any]) -> Optimizer:
        """Return optimizer for the model."""
