from bkd.base.callbacks.abc import EventCallback
from bkd.base.utils.typing.base.pydantic import StandardBaseModel
from bkd.base.utils.typing.events import BatchID, EpochID
from rich.progress import Progress, TaskID

__all__ = ['ProgressCallback']

class ProgressCallback(StandardBaseModel, EventCallback):
    epoch_bar_title: str
    epoch_inner_bar_title: str
    num_epochs: None | int
    progress: Progress
    epoch_task_id: None | TaskID
    current_epoch_inner_id: None | TaskID
    def start(self, num_epochs: None | int = None) -> None: ...
    def log_epoch_start(self, eid: EpochID, epoch_size: None | int = None) -> None: ...
    def log_epoch_end(self, eid: EpochID) -> None: ...
    def log_batch_end(self, bid: BatchID) -> None: ...
    def close_aix(self) -> None: ...
