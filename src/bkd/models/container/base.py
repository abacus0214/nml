"""Base model class."""

from abc import ABC, abstractmethod
from typing import Hashable

from bkd.models.interpreter.base import ModelInterpreter

__all__ = ["ModelABC"]


class ModelABC[InT, OutT, SampleT, LikelihoodT](ABC):
    """Base model class.

    Models should be hashable so that they can be easily
    identified and linked to other (meta)data.
    """

    interpreter: ModelInterpreter[OutT, SampleT, LikelihoodT]

    @abstractmethod
    def forward(self, ipt: InT) -> OutT:
        """Run model prediction."""

    def output(self, ipt: InT) -> OutT:
        """Run forward and pass it to thee interpreter."""
        return self.interpreter.output(self.forward(ipt))

    def ml(self, ipt: InT) -> SampleT:
        """Run forward and then use interpreter to sample."""
        return self.interpreter.ml(self.forward(ipt))

    def sample(self, ipt: InT) -> SampleT:
        """Run forward and then use interpreter to sample."""
        return self.interpreter.sample(self.forward(ipt))

    def log_likelihood(
        self,
        ipt: InT,
        target_samples: SampleT,
    ) -> LikelihoodT:
        """Run forward and then use interpreter to compute the log likelihood."""
        return self.interpreter.log_likelihood(
            model_out=self.forward(ipt), target_samples=target_samples
        )

    @property
    @abstractmethod
    def hash(self) -> Hashable:
        """Return hash of the model."""

    def __hash__(self) -> int:
        """Hash model."""
        return hash(self.hash)
