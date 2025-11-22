"""Base torch Model child class."""

from copy import copy, deepcopy
from typing import Any, Hashable

from nml.models.container.base import ModelABC
from nml.models.interpreter.base import ModelInterpreter
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

__all__ = ["TorchModel"]


class TorchModel[InT, OutT, SampleT](
    RestrictedBaseModel, ModelABC[InT, OutT, SampleT, Tensor]
):
    """Base torch Model child class. Easy containment of torch.nn.Module.

    This class allows to wrap a torch.nn.Module as a nml Model. Also allows the
    user to specify an optimizer to define a custom optimizer to be used for
    this model (some optimizer maps can use this to override default behavior).
    """

    interpreter: ModelInterpreter[OutT, SampleT, Tensor]

    module: Module
    optimizer: None | Optimizer = None

    def forward(self, ipt: InT) -> OutT:
        """Run model prediction."""
        return self.module.forward(ipt)  # type: ignore

    @property
    def hash(self) -> Hashable:
        """Return hash for model."""
        return hash(id(self.module))

    def __deepcopy__(
        self, memo: dict[int, Any] | None = None
    ) -> "TorchModel[InT, OutT, SampleT]":
        """Avoid copying interpreter."""
        # Create copy of module
        module = deepcopy(self.module)

        # Regenerate optimizer if needed
        optimizer: None | Optimizer = None
        if self.optimizer is not None:
            if len(self.optimizer.param_groups) > 1:
                raise NotImplementedError(
                    "Cannot handle partitioned model parameters in optimizer."
                )
            # Shallow copy param groups
            param_groups = [
                copy(param_group) for param_group in self.optimizer.param_groups
            ]
            # Replace parmaeters
            param_groups[0]["params"] = module.parameters()
            # Recreate optimizer
            optimizer = type(self.optimizer)(param_groups, defaults={})

        return TorchModel(
            interpreter=self.interpreter, module=module, optimizer=optimizer
        )
