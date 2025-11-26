"""Buffer storing all events that are passed to it."""

from datetime import datetime
from enum import StrEnum, auto
from types import TracebackType
from typing import Any

from nml.callbacks.stack import CallbackStack
from nml.models.container.base import ModelABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import (
    Figure,
    FigureID,
    Figures,
    FiguresBatch,
    FigureStep,
    Metric,
    MetricID,
    Metrics,
    MetricsBatch,
    MetricStep,
)
from nml.utils.typing.events import BatchID, EpochID
from nml.utils.typing.logging.parameters import ParamsDict
from nml.utils.typing.models.container.base import ModelID
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
    FIGURE = auto()
    FIGURES = auto()
    FIGURE_BATCH = auto()
    CLOSE = auto()
    MODEL = auto()
    PARAMS = auto()


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

    def start_aux(self, params: None | ParamsDict = None) -> None:
        """Call at the start of the process."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.START,
                extra_args={
                    "params": params,
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

    def log_model_aux(
        self,
        model: ModelABC[Any, Any, Any, Any],
        name: None | ModelID = None,
        step: MetricStep = None,
    ) -> None:
        """Log a model."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.MODEL,
                step=step,
                extra_args={"model": model},
            )
        )

    def log_params_aux(
        self,
        params: ParamsDict,
    ) -> None:
        """Call this callback to log a parameter."""

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

    def log_metrics_batch_aux(self, batch: MetricsBatch) -> None:
        """Log multiple metrics samples at arbitrary timesteps at once, by default iterate."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.METRIC_BATCH,
                extra_args={
                    "batch": batch,
                },
            )
        )

    def log_figure_aux(
        self, fid: FigureID, figure: Figure, step: FigureStep = None
    ) -> None:
        """Log figure with given id."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.FIGURE,
                step=step,
                input_id=fid,
                extra_args={
                    "figure": figure,
                },
            )
        )

    def log_figures_aux(self, figures: Figures, step: FigureStep = None) -> None:
        """Log multiple figures at once, by default iterate."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.FIGURES,
                step=step,
                extra_args={
                    "figures": figures,
                },
            )
        )

    def log_figure_batch_aux(self, batch: FiguresBatch) -> None:
        """Log multiple figures samples at arbitrary timesteps at once, by default iterate."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.FIGURE_BATCH,
                extra_args={
                    "batch": batch,
                },
            )
        )

    def close_aux(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> None | bool:
        """Call at the end of the process."""
        self.events_buffer.add(
            CallbackEvent(
                event_type=CallbackEventType.CLOSE,
                extra_args={
                    "exc_type": exc_type,
                    "exc_val": exc_val,
                    "exc_tb": exc_tb,
                },
            )
        )
        return False
