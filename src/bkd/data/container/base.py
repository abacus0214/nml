"""Base class for data container."""

from typing import Iterable

__all__ = ["Dataset"]

type Dataset[BatchT] = Iterable[BatchT]
