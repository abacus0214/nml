"""Maximum likelihood loss runners."""

from typing import Any

from nml.data.batches.container.base import BatchABC
from nml.loss.runner.base import LossRunnerABC
from nml.models.container.base import ModelABC

__all__ = ["MLLossRunner"]


class MLLossRunner(LossRunnerABC):
    """Maxumum likelihood loss."""

    def compute[IpT, TgT, LossT](
        self,
        batch: BatchABC[Any, IpT, TgT],
        model: ModelABC[IpT, Any, TgT, LossT],
    ) -> LossT:
        """Compute the loss from a given model and batch."""
        return model.nll(ipt=batch.ipt, target_samples=batch.tgt)
