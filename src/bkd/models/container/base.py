"""Base model class."""

from abc import ABC, abstractmethod

from bkd.models.interpreter.base import ModelInterpreter

__all__ = ["ModelABC"]


class ModelABC[InT, OutT, SampleT, LikelihoodT](ABC):
    """Base model class."""

    interpreter: ModelInterpreter[OutT, SampleT, LikelihoodT]

    @abstractmethod
    def forward(self, ipt: InT) -> OutT:
        """Run model prediction."""

    def pred(self, ipt: InT) -> OutT:
        """Run forward and pass it to thee interpreter."""
        return self.interpreter.pred(self.forward(ipt))

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
