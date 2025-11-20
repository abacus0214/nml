"""Some standard frequencers."""

from nml.tools.utils.frequencer.base import FrequencerABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.events import EpochID, TrainingStepID

__all__ = ["AtStart", "AtEnd", "Every", "At"]


class AtStart(FrequencerABC):
    """Only perform operation at the start."""

    def __call__(self, step_id: TrainingStepID) -> bool:
        """Return `True` if operation should be performed."""
        return step_id.eid == 0


class AtEnd(FrequencerABC):
    """Only perform operation at the end."""

    def __call__(self, step_id: TrainingStepID) -> bool:
        """Return `True` if operation should be performed."""
        return step_id.eid == step_id.eid_max


class Every(RestrictedBaseModel, FrequencerABC):
    """Only perform operation at the end."""

    freq: int

    def __call__(self, step_id: TrainingStepID) -> bool:
        """Return `True` if operation should be performed."""
        return (step_id.eid % self.freq) == 0


class At(RestrictedBaseModel, FrequencerABC):
    """Only perform operation at the end."""

    eids: set[EpochID]

    def __call__(self, step_id: TrainingStepID) -> bool:
        """Return `True` if operation should be performed."""
        return step_id.eid in self.eids
