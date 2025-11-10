from mlflow.entities import Dataset as MLFlowDataset, Metric as MLFlowMetric, Param as MLFlowParam, Run as MLFlowRun, RunTag as MLFlowRunTag
from typing import Any

__all__ = ['MLFlowMetric', 'MLFlowParam', 'MLFlowRunTag', 'MLFlowTagsDict', 'MLFlowDataset', 'MLFlowRun']

type MLFlowTagsDict = dict[str, Any]
