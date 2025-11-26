"""Base Torch interpreter."""

from abc import ABC, abstractmethod

from nml.models.interpreter.base import ModelInterpreter
from torch import Tensor
from torch.distributions import Distribution

__all__ = ["TorchModelInterpreter", "TorchDistInterpreter"]


class TorchModelInterpreter[ModelOutT](ModelInterpreter[ModelOutT, Tensor, Tensor]):
    """Parent class of all torch models interpreter."""


class TorchDistInterpreter[ModelOutT](ABC, TorchModelInterpreter[ModelOutT]):
    """Premade interpreter wrapper from torch distro."""

    @abstractmethod
    def build_dist(self, model_out: ModelOutT) -> Distribution:
        """Define how to construct a torch.distribution.Distribution type object from the model output."""

    def sample(self, model_out: ModelOutT) -> Tensor:
        """Get one sample based on the model output.

        This method should be stochastic and should sample from the distribution
        that the mdoel has given as output.

        Even though this is not an abstract method it should be implemented when
        defining a model interpreter. The default implementation allows for
        interpreters to only have a subsets of the methods implemented, which is
        fine as long as the user understands that this is the case.
        """
        return self.build_dist(model_out=model_out).sample()

    def ml(self, model_out: ModelOutT) -> Tensor:
        """Get the maximum likelihood sample.

        This method is what is commonly referred to as the model prediction, which
        most of the time is the maximum likelihood sample from the model's output
        distribution.

        Even though this is not an abstract method it should be implemented when
        defining a model interpreter. The default implementation allows for
        interpreters to only have a subsets of the methods implemented, which is
        fine as long as the user understands that this is the case.
        """
        return self.build_dist(model_out=model_out).mode

    def nll(self, target_samples: Tensor, model_out: ModelOutT) -> Tensor:
        """Compute the log likelihood of a given set of sample based on the model output.

        By default, explicitly compute the log_prob from the Distribution class. However,
        it is advised to overwrite this with the already implemented loss functions from
        pytorch for any particular class.
        """
        return self.build_dist(model_out=model_out).log_prob(target_samples).sum()
