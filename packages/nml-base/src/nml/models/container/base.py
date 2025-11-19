"""Base model class."""

from abc import ABC, abstractmethod
from typing import Hashable

from nml.models.interpreter.base import ModelInterpreter, PredictionType

__all__ = ["ModelABC"]


class ModelABC[InT, OutT, SampleT, LikelihoodT](ABC):
    """Base model class.

    This wrapper should pair a particular model (thorugh the
    forward method) with a specific intepreter, following the more
    classical wrapper pattern. The methods in this class just
    forward to methods of the interpeter by passing the output
    of `self.forward` as the provided model output.

    Models should be hashable so that they can be easily
    identified and linked to other (meta)data.
    """

    interpreter: ModelInterpreter[OutT, SampleT, LikelihoodT]

    @abstractmethod
    def forward(self, ipt: InT) -> OutT:
        """Run model prediction."""

    def inference(self, ipt: InT) -> OutT:
        """Run model prediction (for evaluation)."""
        return self.interpreter.inference(model_out=self.forward(ipt=ipt))

    def predict(
        self, ipt: InT, tp: PredictionType = PredictionType.OUTPUT
    ) -> OutT | SampleT:
        """Forward to interpreter.__call__."""
        return self.interpreter(model_out=self.forward(ipt=ipt), tp=tp)

    def output(self, ipt: InT) -> OutT:
        """Run forward and pass it to thee interpreter."""
        return self.interpreter.output(self.forward(ipt))

    def ml(self, ipt: InT) -> SampleT:
        """Run forward and then use interpreter to sample."""
        return self.interpreter.ml(self.forward(ipt))

    def sample(self, ipt: InT) -> SampleT:
        """Run forward and then use interpreter to sample."""
        return self.interpreter.sample(self.forward(ipt))

    def nll(
        self,
        ipt: InT,
        target_samples: SampleT,
    ) -> LikelihoodT:
        """Run forward and then use interpreter to compute the log likelihood."""
        return self.interpreter.nll(
            model_out=self.forward(ipt), target_samples=target_samples
        )

    @property
    @abstractmethod
    def hash(self) -> Hashable:
        """Return what should be used to hash the model."""

    def __hash__(self) -> int:
        """Hash model."""
        return hash(self.hash)
