import abc
from abc import ABC, abstractmethod
from bkd.base.data.batches.container.base import BatchABC
from bkd.base.loss.runner.base import LossRunnerABC
from bkd.base.models.container.base import ModelABC
from typing import Any

__all__ = ['BatchLossTrainerABC']

class BatchLossTrainerABC[LossT](ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def batch_loss[IpT, TgT](self, model: ModelABC[IpT, Any, TgT, LossT], batch: BatchABC[Any, IpT, TgT], loss_runner: LossRunnerABC, update_model: bool = False) -> LossT: ...
