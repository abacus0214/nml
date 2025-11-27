"""Standard types for timestep samples in replay buffer."""

from typing import NamedTuple

from nml.rl.utils.types.events.base import EpisodeID, TimeID

__all__ = ["TimestepSampleBatch", "TimestepSampleLink"]


class TimestepSampleBatch[BatchT](NamedTuple):
    """Tuple containing single timestep sample."""

    batch: BatchT

    episode: EpisodeID
    time: TimeID

    next_sample_at: None | TimeID = None


class TimestepSampleLink[BatchT](NamedTuple):
    """Link two timestep sampels of the same type."""

    sample: TimestepSampleBatch[BatchT]
    succ: TimestepSampleBatch[BatchT]
