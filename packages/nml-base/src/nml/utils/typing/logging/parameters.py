"""Types to be used for parameters."""

from typing import Any

__all__ = [
    "ParamID",
    "ParamVal",
    "ParamsDict",
]

type ParamID = str
type ParamVal = Any

type ParamsDict = dict[ParamID, ParamVal]
