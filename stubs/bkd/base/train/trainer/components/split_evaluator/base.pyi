from bkd.base.data.loader.container.base import DataLoaderABC
from bkd.base.models.container.base import ModelABC
from bkd.base.utils.typing.eval.metrics import Metrics
from typing import Any

__all__ = ['SplitEvaluator']

class SplitEvaluator:
    def evaluate_split[IpT, TgT](self, model: ModelABC[IpT, Any, TgT, Any], dataset: DataLoaderABC[Any, IpT, TgT]) -> Metrics: ...
