"""Useful types to be used in general model interpreters."""

from enum import StrEnum, auto


class InterpreterMethod(StrEnum):
    """Enum representing a particular interpreter method."""

    PRED = auto()
    SAMPLE = auto()
    LOG_LIKELIHOOD = auto()
