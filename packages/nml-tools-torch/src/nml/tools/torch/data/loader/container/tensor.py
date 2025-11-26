"""Base class for data container."""

from nml.data.batches.interpreter.base import BatchInterpreter
from nml.data.batches.interpreter.standard import TupleBatchInterpreter
from nml.itf.torch.data.loader.container.base import TorchDataLoader
from pydantic import Field

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["TensorDataLoader"]


class TensorDataLoader(TorchDataLoader[tuple[Tensor, Tensor], Tensor, Tensor]):
    """Premade data loader container for array dataeset."""

    batch_interpreter: BatchInterpreter[tuple[Tensor, Tensor], Tensor, Tensor] = Field(
        default_factory=TupleBatchInterpreter[Tensor, Tensor]
    )

    @staticmethod
    def from_torch_dataset(
        dataset: TensorDataset, batch_size: int = 1, shuffle: bool = True
    ) -> "TensorDataLoader":
        """Construct from a dataset instead of a given dataloader."""
        return TensorDataLoader(
            torch_loader=DataLoader[tuple[Tensor, Tensor]](
                dataset,  # type: ignore
                batch_size=batch_size,
                shuffle=shuffle,
            ),
        )
