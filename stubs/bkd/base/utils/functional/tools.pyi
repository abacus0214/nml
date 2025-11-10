from typing import Callable

__all__ = ['compose_pair']

def compose_pair[XT, IT, OT](f1: Callable[[XT], IT], f2: Callable[[IT], OT]) -> Callable[[XT], OT]: ...
