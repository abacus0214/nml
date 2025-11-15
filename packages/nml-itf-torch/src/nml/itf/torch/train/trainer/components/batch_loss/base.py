"""Batch loss trainer for torch models."""

from typing import Any

from nml.data.batches.container.base import BatchABC
from nml.itf.torch.train.trainer.components.optimizer_map.base import OptimizerMapABC
from nml.loss.runner.base import LossRunnerABC
from nml.models.container.base import ModelABC
from nml.train.trainer.components.batch_loss.base import BatchLossTrainerABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from torch import Tensor

__all__ = ["TorchBatchLossTrainer"]


class TorchBatchLossTrainer(RestrictedBaseModel, BatchLossTrainerABC[Tensor]):
    """Base class for batch loss trainer."""

    optimizer_map: OptimizerMapABC

    def batch_loss[IpT, TgT](
        self,
        model: ModelABC[IpT, Any, TgT, Tensor],
        batch: BatchABC[Any, IpT, TgT],
        loss_runner: LossRunnerABC,
        update_model: bool = False,
    ) -> Tensor:
        """Compute loss and update model (if required)."""
        # Compute loss
        loss = loss_runner.compute(batch=batch, model=model)

        # Perform update if necessary
        if update_model:
            self.optimizer_map.get_optimizer(model=model).zero_grad()
            loss.backward()
            self.optimizer_map.get_optimizer(model=model).step()

        return loss
