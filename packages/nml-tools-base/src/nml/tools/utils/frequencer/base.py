"""Useful object to define frequencies."""

from abc import ABC, abstractmethod

from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.events import TrainingStepID

__all__ = ["FrequencerABC", "ANDFrequencer", "ORFrequencer"]


class FrequencerABC(ABC):
    """Simple object to determine the frequencies of operations."""

    @abstractmethod
    def __call__(self, step_id: TrainingStepID) -> bool:
        """Return `True` if operation should be performed."""

    def __and__(self, other: "FrequencerABC") -> "FrequencerABC":
        """Connect two frequencers with conjunction."""
        return ANDFrequencer(frequencers=(self, other))

    def __or__(self, other: "FrequencerABC") -> "FrequencerABC":
        """Connect two frequencers with disjunction."""
        return ORFrequencer(frequencers=(self, other))


class ANDFrequencer(RestrictedBaseModel, FrequencerABC):
    """Conjuction of multiple frequencers."""

    frequencers: tuple[FrequencerABC, ...]

    def __call__(self, step_id: TrainingStepID) -> bool:
        """Return `True` if operation should be performed."""
        return all((freq(step_id) for freq in self.frequencers))

    def __and__(self, other: "FrequencerABC") -> "FrequencerABC":
        """Connect two frequencers with conjunction."""
        return ANDFrequencer(frequencers=self.frequencers + (other,))


class ORFrequencer(RestrictedBaseModel, FrequencerABC):
    """Disjunctions of multiple frequencers."""

    frequencers: tuple[FrequencerABC, ...]

    def __call__(self, step_id: TrainingStepID) -> bool:
        """Return `True` if operation should be performed."""
        return any((freq(step_id) for freq in self.frequencers))

    def __or__(self, other: "FrequencerABC") -> "FrequencerABC":
        """Connect two frequencers with disjunction."""
        return ORFrequencer(frequencers=self.frequencers + (other,))
