"""DataLoader wrapper for torch.utils.data.DataLoader class."""

from typing import Iterator

from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoaderAux

from bkd.data.loader.container.base import DataLoaderABC


class TorchDataLoader[BatchT, IpT: Tensor, TgT: None | Tensor = None](
    DataLoaderABC[BatchT, IpT, TgT]
):
    """Base data loader container."""

    name: str
    torch_loader: TorchDataLoaderAux[BatchT]

    @property
    def epoch_iterator(self) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""
        return iter(self.torch_loader)
