"""Basic types for events in RL."""

from typing import Hashable

__all__ = ["EpisodeID", "TimeID"]

type EpisodeID = Hashable
type TimeID = float
