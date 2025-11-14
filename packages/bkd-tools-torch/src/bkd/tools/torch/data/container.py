"""Implementations of torch.utils.data.Dataset abstract class."""

from typing import Any

from numpy import dtype, float32, ndarray
from torch.utils.data import Dataset

__all__ = ["ArrayDataset", "FloatArray"]

type FloatArray = ndarray[Any, dtype[float32]]


class ArrayDataset(Dataset[FloatArray]):
    """Custom dataset that gets matrix data and turns it into a torch dataset."""

    X: FloatArray
    y: FloatArray

    def __init__(self, X: FloatArray, y: FloatArray) -> None:
        """Construct."""
        super().__init__()

        # Check shape of input data
        if X.shape[0] != y.shape[0]:
            raise TypeError(f"Mismatching dimensions {X.shape[0]} and {y.shape[0]}")

        # Store input data
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """Compute length."""
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[FloatArray, FloatArray]:  # type: ignore
        """Sample a single element."""
        return self.X[idx], self.y[idx]
