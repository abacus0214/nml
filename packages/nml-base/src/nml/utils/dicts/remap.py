"""Basic functions to edit mappings."""

from collections import UserDict
from typing import Mapping

__all__ = ["add_prefix"]


def add_prefix[VT](mapping: Mapping[str, VT], prefix: str) -> Mapping[str, VT]:
    """Add prefix to all keys in a mapping."""
    # Remap values and turn to dict
    remapped = {f"{prefix}/{key}": val for key, val in mapping.items()}

    if not isinstance(mapping, UserDict):
        return remapped

    # Keep the same UserDict class if possible
    try:
        return mapping.__class__(remapped)
    except:
        return remapped
