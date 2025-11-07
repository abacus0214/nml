"""Base class for data container."""

from typing import Iterable

__all__ = ["DataLoader"]

type DataLoader[BatchT] = Iterable[BatchT]
