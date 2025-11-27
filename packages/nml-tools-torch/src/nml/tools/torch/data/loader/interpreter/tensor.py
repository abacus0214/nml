"""Base class for data container."""

from nml.data.batches.interpreter.base import BatchInterpreter
from nml.data.batches.interpreter.standard import TupleBatchInterpreter
from nml.itf.torch.data.loader.interpreter.base import TorchLoaderInterpreter
from pydantic import Field

from torch import Tensor

__all__ = ["TensorLoaderInterpreter"]


class TensorLoaderInterpreter(
    TorchLoaderInterpreter[tuple[Tensor, Tensor], Tensor, Tensor]
):
    """Premade data loader container for array dataeset."""

    batch_interpreter: BatchInterpreter[tuple[Tensor, Tensor], Tensor, Tensor] = Field(
        default_factory=TupleBatchInterpreter[Tensor, Tensor]
    )
