"""Buffer storing all events that are passed to it."""

from datetime import datetime
from enum import StrEnum, auto
from typing import Any

from nml.callbacks.stack import CallbackStack
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import (
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from nml.utils.typing.events import BatchID, EpochID
from pydantic import Field

__all__ = ["CallbackStack", "CallbackEventType", "CallbackEvent", "EventsBuffer"]


class CallbackEventType(StrEnum):
    """Types of events that a callback can experience."""

    START = auto()
    EPOCH_START = auto()
    EPOCH_END = auto()
    BATCH_SART = auto()
    BATCH_END = auto()
    METRIC = auto()
    METRICS = auto()
    METRIC_BATCH = auto()
    CLOSE = auto()


class CallbackEvent(RestrictedBaseModel):
    """Record a callback event."""

    event_type: CallbackEventType
    step: None | int = None
    input_id: None | str | int = None
    extra_args: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


type EventsBuffer = set[CallbackEvent]


class CallbackBuffer(RestrictedBaseModel, CallbackStack):
    """Record all events."""

    events_buffer: EventsBuffer = Field(default_factory=set)

    def start_aux(self, num_epochs: None | int = None) -> None:
        """Call at the start of the process."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.START,
                extra_args={
                    "num_epochs": num_epochs,
                },
            )
        )

    def log_epoch_start_aux(self, eid: EpochID, epoch_size: None | int = None) -> None:
        """Call this callback when epoch ends."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.EPOCH_START,
                input_id=eid,
                extra_args={
                    "epoch_size": epoch_size,
                },
            )
        )

    def log_epoch_end_aux(self, eid: EpochID) -> None:
        """Call this callback when epoch ends."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.EPOCH_END,
                input_id=eid,
            )
        )

    def log_batch_start_aux(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.BATCH_SART,
                input_id=bid,
            )
        )

    def log_batch_end_aux(self, bid: BatchID) -> None:
        """Call this callback when batch ends."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.BATCH_END,
                input_id=bid,
            )
        )

    def log_metric_aux(
        self, mid: MetricID, metric: Metric, step: MetricStep = None
    ) -> None:
        """Log metric with given id."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.METRIC,
                step=step,
                input_id=mid,
                extra_args={
                    "metric": metric,
                },
            )
        )

    def log_metrics_aux(self, metrics: Metrics, step: MetricStep = None) -> None:
        """Log multiple metrics at once, by default iterate."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.METRICS,
                step=step,
                extra_args={
                    "metrics": metrics,
                },
            )
        )

    def log_batch_aux(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.METRIC_BATCH,
                extra_args={
                    "batch": batch,
                },
            )
        )

    def close_aux(self) -> None:
        """Call at the end of the process."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.CLOSE,
            )
        )
