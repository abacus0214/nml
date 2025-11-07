"""Base class for model interpreter for torch models."""

from torch import Tensor

from bkd.models.interpreter.base import ModelInterpreter

__all__ = ["TorchModelInterpreter"]


class TorchModelInterpreter[ParamsT, SampleT](
    ModelInterpreter[ParamsT, SampleT, Tensor]
):
    """Parent class for all torch model interpreters."""
