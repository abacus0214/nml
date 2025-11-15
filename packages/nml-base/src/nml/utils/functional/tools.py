"""Useful functools utilities."""

from typing import Callable

__all__ = ["compose_pair"]


def compose_pair[XT, IT, OT](
    f1: Callable[[XT], IT], f2: Callable[[IT], OT]
) -> Callable[[XT], OT]:
    """Compose two functions."""

    def composed(x: XT) -> OT:
        """Composition of two functions."""
        return f2(f1(x))

    return composed
