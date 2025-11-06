"""Metric callback with progress bar."""

from pydantic import Field
from rich.progress import Progress, TaskID

from bkd.eval.metrics.callback.stack import CallbackStack
from bkd.utils.typing.base.pydantic import StandardBaseModel
from bkd.utils.typing.events import BatchID, EpochID

__all__ = ["ProgressCallback"]


class ProgressCallback(StandardBaseModel, CallbackStack):
    """Have progress bar for each epoch."""

    # Required parameters
    num_epochs: int
    epoch_size: int

    # UI config
    epoch_bar_title: str = "Epochs"
    epoch_inner_bar_title: str = "Running Epoch[{eid}]"

    # Memory
    progress: Progress = Field(default_factory=Progress, init=False)
    epoch_task_id: None | TaskID = Field(init=False, default=None)
    current_epoch_inner_id: None | TaskID = Field(init=False, default=None)

    def log_epoch_start_aux(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""
        # If epoch task id has not been created,
        # do so and start progress bar
        if self.epoch_task_id is None:
            self.progress.start()
            self.epoch_task_id = self.progress.add_task(
                self.epoch_bar_title, total=self.num_epochs - 1
            )

        self.current_epoch_inner_id = self.progress.add_task(
            self.epoch_inner_bar_title.format(eid=eid), total=self.epoch_size - 1
        )

    def log_epoch_end_aux(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""
        if self.epoch_task_id is None:
            raise RuntimeError("No epoch started.")

        # Increment epoch progress bar
        self.progress.update(self.epoch_task_id, advance=1, update=True)

    def log_batch_end_aux(self, bid: BatchID) -> None:
        """Incremet inner progress bar."""
        if self.current_epoch_inner_id is None:
            raise RuntimeError("No epoch started.")
        self.progress.update(self.current_epoch_inner_id, advance=1, update=True)

    def close_aix(self) -> None:
        """Call at end to close progress bar."""
        self.progress.refresh()
        self.progress.stop()
        self.progress.console.clear_live()
