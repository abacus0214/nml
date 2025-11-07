"""Standard pytorch trainer."""

from torch.nn import Module
from torch.utils.data import DataLoader

from bkd.eval.metrics.callback.abc import MetricCallback
from bkd.itf.torch.train.trainer.base import TorchTrainer


class StandardTorchtrainer[BatchT](TorchTrainer[BatchT]):
    """Base class that defines the interface for a trainer."""

    def train(
        self,
        model: Module,
        dataset: DataLoader[BatchT],
        callback: MetricCallback = MetricCallback(),
    ) -> None:
        """Perform training loop."""
