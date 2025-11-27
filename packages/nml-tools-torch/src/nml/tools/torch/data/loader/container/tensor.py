"""Routines to build dataset loader containers for torch datasets."""

from nml.data.loader.container.base import LoaderContainer
from nml.data.loader.interpreter.base import LoaderInterpreter
from nml.tools.torch.data.loader.interpreter.tensor import TensorLoaderInterpreter
from pydantic import Field

from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

__all__ = ["TensorLoaderContainer"]


class TensorLoaderContainer(
    LoaderContainer[
        TorchDataLoader[tuple[Tensor, Tensor]], tuple[Tensor, Tensor], Tensor, Tensor
    ]
):
    """Container for Tensor datasets."""

    loader_interpreter: LoaderInterpreter[
        TorchDataLoader[tuple[Tensor, Tensor]], tuple[Tensor, Tensor], Tensor, Tensor
    ] = Field(default_factory=TensorLoaderInterpreter)

    @staticmethod
    def from_torch_dataset(
        dataset: TensorDataset, batch_size: int = 1, shuffle: bool = True
    ) -> "TensorLoaderContainer":
        """Construct from a dataset instead of a given dataloader."""
        return TensorLoaderContainer(
            loader=TorchDataLoader[tuple[Tensor, Tensor]](
                dataset,  # type: ignore
                batch_size=batch_size,
                shuffle=shuffle,
            ),
        )
