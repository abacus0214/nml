"""Useful type aliases for evaluation events."""

from typing import NamedTuple

from nml.utils.typing.data.dataset import DatasetID

__all__ = ["EpochID", "BatchID", "TrainingStepID"]

type EpochID = int
type BatchID = int


class TrainingStepID(NamedTuple):
    """Tuple representing one specific training instance.

    This should encompass the name of the dataset that has
    generated the batch, the current epoch, the current batch
    id, and optionally the maximum epoch allowed
    """

    did: DatasetID

    bid: BatchID

    eid: EpochID
    eid_max: None | EpochID = None
