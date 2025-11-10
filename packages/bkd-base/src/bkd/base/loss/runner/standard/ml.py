"""Maximum likelihood loss runners."""

from typing import Any

from bkd.base.data.batches.container.base import BatchABC
from bkd.base.loss.runner.base import LossRunnerABC
from bkd.base.models.container.base import ModelABC

__all__ = ["MLLossRunner"]


class MLLossRunner(LossRunnerABC):
    """Maxumum likelihood loss."""

    def compute[IpT, TgT, LossT](
        self,
        batch: BatchABC[Any, IpT, TgT],
        model: ModelABC[IpT, Any, TgT, LossT],
    ) -> LossT:
        """Compute the loss from a given model and batch."""
        return model.log_likelihood(ipt=batch.ipt, target_samples=batch.tgt)
