"""Useful types and type aliases for mlflow."""

from typing import Any, TypeAlias

from mlflow.entities import Run

__all__ = [
    "Run",
    "MLFlowStepType",
    "MLFlowExperimentID",
    "MLFlowTimeType",
    "MLFlowTagsDict",
    "MLFlowRunName",
    "MLFlowRunID",
    "MLFlowMetricKey",
    "MLFlowMetricVal",
    "MLFlowModelID",
    "MLFlowDatasetDigest",
    "MLFlowDatasetName",
]

# Generatl General
MLFlowTimeType: TypeAlias = int
MLFlowStepType: TypeAlias = int

# Expriments
MLFlowExperimentID: TypeAlias = str

# Runs
MLFlowTagsDict: TypeAlias = dict[str, Any]
MLFlowRunName: TypeAlias = str
MLFlowRunID: TypeAlias = str

# Metrics
MLFlowMetricKey: TypeAlias = str
MLFlowMetricVal: TypeAlias = Any

# Model
MLFlowModelID: TypeAlias = str

# Dataset
MLFlowDatasetName: TypeAlias = str
MLFlowDatasetDigest: TypeAlias = str
