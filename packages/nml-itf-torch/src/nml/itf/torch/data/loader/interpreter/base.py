"""DataLoader wrapper for torch.utils.data.DataLoader class."""

from typing import Any, Iterator

from nml.data.batches.interpreter.base import BatchInterpreter
from nml.data.loader.interpreter.base import LoaderInterpreter
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader

__all__ = ["TorchLoaderInterpreter"]


class TorchLoaderInterpreter[BatchT, IpT: Tensor, TgT: None | Tensor = None](
    RestrictedBaseModel, LoaderInterpreter[TorchDataLoader[Any], BatchT, IpT, TgT]
):
    """Base data loader container."""

    batch_interpreter: BatchInterpreter[BatchT, IpT, TgT]

    def raw_epoch_iterator(self, dataset: TorchDataLoader[Any]) -> Iterator[BatchT]:
        """Return iterator to go through batches for a single epoch."""
        return iter(dataset)

    def len(self, dataset: TorchDataLoader[Any]) -> int:
        """Get length from loader."""
        return len(dataset)
