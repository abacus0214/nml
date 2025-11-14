"""Base Torch interpreter."""

from bkd.models.interpreter.base import ModelInterpreter
from torch import Tensor

__all__ = ["TorchModelInterpreter"]


class TorchModelInterpreter[SampleT](ModelInterpreter[Tensor, SampleT, Tensor]):
    """Parent class of all torch models interpreter."""
