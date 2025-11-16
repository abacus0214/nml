"""Interpreter for output layer estimating the mean of a normal distribution."""

from nml.itf.torch.models.interpreter.base import TorchDistInterpreter
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from pydantic import Field
from torch import Tensor
from torch.distributions.normal import Normal
from torch.nn.functional import gaussian_nll_loss, mse_loss

__all__ = ["TorchNormalInterpeter", "NormalOutType"]

type NormalOutType = Tensor | tuple[Tensor, Tensor]


class TorchNormalInterpeter(RestrictedBaseModel, TorchDistInterpreter[NormalOutType]):
    """Pre packaged interpreter for regression output layers."""

    default_scale: Tensor = Field(default_factory=lambda: Tensor([1.0]))

    def build_dist(self, model_out: NormalOutType) -> Normal:
        """Define how to construct a torch.distribution.Distribution type object from the model output."""
        # If scale is not specified, use default value
        scale = self.default_scale

        # Parse the output of the model
        if isinstance(model_out, tuple):
            loc, scale = model_out
        else:
            loc = model_out

        # Build distribution
        return Normal(loc=loc, scale=scale)

    def nll(self, target_samples: Tensor, model_out: NormalOutType) -> Tensor:
        """Compute the log likelihood of a given set of sample based on the model output.

        By default, explicitly compute the log_prob from the Distribution class. However,
        it is advised to overwrite this with the already implemented loss functions from
        pytorch for any particular class.
        """
        if isinstance(model_out, tuple):
            mean, var = model_out
            return gaussian_nll_loss(mean, target_samples, var)

        return mse_loss(model_out, target_samples)
