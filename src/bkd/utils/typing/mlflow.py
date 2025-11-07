"""Useful types and type aliases for mlflow."""

from typing import Any, TypeAlias

from mlflow.entities import Dataset as MLFlowDataset
from mlflow.entities import Metric as MLFlowMetric
from mlflow.entities import Param as MLFlowParam
from mlflow.entities import Run as MLFlowRun
from mlflow.entities import RunTag as MLFlowRunTag

__all__ = [
    "MLFlowMetric",
    "MLFlowParam",
    "MLFlowRunTag",
    "MLFlowTagsDict",
    "MLFlowDataset",
    "MLFlowRun",
]

# Runs
MLFlowTagsDict: TypeAlias = dict[str, Any]
