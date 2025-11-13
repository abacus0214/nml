"""Utility operator."""

from operator import itemgetter

__all__ = ["itemgetter", "identity"]


def identity[T](x: T) -> T:
    """Identity function."""
    return x
