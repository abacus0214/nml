"""Base class for data container."""

from nml.data.batches.interpreter.base import BatchInterpreter
from nml.data.batches.interpreter.standard import TupleBatchInterpreter
from nml.itf.torch.data.loader.container.base import TorchDataLoader
from nml.tools.torch.data.dataset.array import ArrayDataset
from pydantic import Field

from torch import Tensor
from torch.utils.data import DataLoader

__all__ = ["ArrayDataLoader"]


class ArrayDataLoader(TorchDataLoader[tuple[Tensor, Tensor], Tensor, Tensor]):
    """Premade data loader container for array dataeset."""

    batch_interpreter: BatchInterpreter[tuple[Tensor, Tensor], Tensor, Tensor] = Field(
        default_factory=TupleBatchInterpreter[Tensor, Tensor]
    )

    @staticmethod
    def from_torch_dataset(
        name: str, dataset: ArrayDataset, batch_size: int = 1
    ) -> "ArrayDataLoader":
        """Construct from a dataset instead of a given dataloader."""
        return ArrayDataLoader(
            name=name,
            torch_loader=DataLoader[tuple[Tensor, Tensor]](
                dataset, batch_size=batch_size
            ),
        )
