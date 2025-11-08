"""Base class for loss runners."""

from abc import ABC, abstractmethod

from bkd.models.interpreter.base import ModelInterpreter

__all__ = ["LossRunner"]


class LossRunner[LossT](ABC):
    """Wrapper for callacble that computes a loss based on a batch and model."""

    @abstractmethod
    def __call__(
        self,
        batch: object,
        model: object,
        interpreter: ModelInterpreter[object, object, LossT],
    ) -> LossT:
        """Compute the loss from a given model and batch."""
