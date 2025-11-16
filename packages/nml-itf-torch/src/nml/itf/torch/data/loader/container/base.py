"""DataLoader wrapper for torch.utils.data.DataLoader class."""

from typing import Any, Iterator

from nml.data.batches.interpreter.base import BatchInterpreter
from nml.data.loader.container.base import DataLoaderABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoaderAux


class TorchDataLoader[BatchT, IpT: Tensor, TgT: None | Tensor = None](
    RestrictedBaseModel, DataLoaderABC[BatchT, IpT, TgT]
):
    """Base data loader container."""

    name: str
    torch_loader: TorchDataLoaderAux[Any]  # TODO: fix this type hint

    batch_interpreter: BatchInterpreter[BatchT, IpT, TgT]

    @property
    def raw_epoch_iterator(self) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""
        return iter(self.torch_loader)

    def __len__(self) -> int:
        """Get length from loader."""
        return len(self.torch_loader)
