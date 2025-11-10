from bkd.base.data.batches.container.base import BatchABC
from bkd.base.loss.runner.base import LossRunnerABC
from bkd.base.models.container.base import ModelABC
from typing import Any

__all__ = ['MLLossRunner']

class MLLossRunner(LossRunnerABC):
    def compute[IpT, TgT, LossT](self, batch: BatchABC[Any, IpT, TgT], model: ModelABC[IpT, Any, TgT, LossT]) -> LossT: ...
