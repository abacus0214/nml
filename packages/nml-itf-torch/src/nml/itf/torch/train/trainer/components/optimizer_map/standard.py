"""Standard implementation of OptimizerMapABC."""

from collections import UserDict
from typing import Any

from nml.itf.torch.models.container.base import TorchModel
from nml.itf.torch.train.trainer.components.optimizer_map.base import OptimizerMapABC
from nml.models.container.base import ModelABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from pydantic import Field
from torch.optim import Adam, Optimizer

__all__ = [
    "TorchModelOptimizerMap",
    "DictOptimizerMap",
    "OptimizerNotFound",
    "DefaultDictOptimizerMap",
    "get_standard_optimizer_map",
]


class OptimizerNotFound(RuntimeError):
    """Exception to be raised when optimizer cannot be established."""

    def __init__(self, model: ModelABC[Any, Any, Any, Any]) -> None:
        """Set default message."""
        super().__init__(
            f"Optimizer could not be found or constructed for model {model}"
        )


class TorchModelOptimizerMap(OptimizerMapABC):
    """Map from model to optimizer specific for torch model."""

    def get_optimizer(self, model: ModelABC[Any, Any, Any, Any]) -> Optimizer:
        """Return optimizer for the model."""
        # Initialize variable in which to store optimizer
        optimizer: None | Optimizer = None
        # Use optimizer from torch model if available
        if isinstance(model, TorchModel):
            optimizer = model.optimizer

        # If optimizer is None because the model is not a torch model
        # or because the torch model does not have an optimizer set
        # then raise an exception no matter what
        if optimizer is None:
            raise OptimizerNotFound(model=model)

        return optimizer


class DictOptimizerMap(
    UserDict[ModelABC[Any, Any, Any, Any], Optimizer], OptimizerMapABC
):
    """Map for model optimizer that will use the hash of the model to map it to an optimizer."""

    optimizer_cls: type[Optimizer] = Adam
    optimizer_args: dict[str, Any]

    def __init__(
        self,
        data: dict[ModelABC[Any, Any, Any, Any], Optimizer] = {},
        *,
        optimizer_cls: type[Optimizer] = Adam,
        **optimizer_args: Any,
    ) -> None:
        """Store arguments."""
        UserDict.__init__(self, data)

        # Store parameters to generate optimizers.
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args

    def get_optimizer(self, model: ModelABC[Any, Any, Any, Any]) -> Optimizer:
        """Return optimizer for the model."""
        return self[model]

    def __missing__(self, key: ModelABC[Any, Any, Any, Any]) -> Optimizer:
        """In case of missing optimizer, build one."""
        # Add new optimizer if not done so yet.
        if key not in self.keys():
            self[key] = self.construct_optimizer(model=key)
        return self[key]

    def construct_optimizer(self, model: ModelABC[Any, Any, Any, Any]) -> Optimizer:
        """Construct the optimizer for the given model."""
        # Only works for torch models.
        if not isinstance(model, TorchModel):
            raise OptimizerNotFound(model=model)

        return self.optimizer_cls(model.module.parameters(), **self.optimizer_args)


class DefaultDictOptimizerMap(RestrictedBaseModel, OptimizerMapABC):
    """Dict optimizer map, but if the torch model specifies an optimizer by itself, then that is used in its place."""

    dict_optimizer_map: DictOptimizerMap
    torch_optimizer_map: TorchModelOptimizerMap = Field(
        default_factory=TorchModelOptimizerMap
    )

    def get_optimizer(self, model: ModelABC[Any, Any, Any, Any]) -> Optimizer:
        """Return optimizer for the model."""
        try:
            return self.torch_optimizer_map.get_optimizer(model=model)
        except OptimizerNotFound:
            return self.dict_optimizer_map.get_optimizer(model=model)


def get_standard_optimizer_map(
    optimizer_cls: type[Optimizer] = Adam,
    **optimizer_args: Any,
) -> DefaultDictOptimizerMap:
    """Syntax sugar to build the standard optimizer."""
    return DefaultDictOptimizerMap(
        dict_optimizer_map=DictOptimizerMap(
            optimizer_cls=optimizer_cls,
            **optimizer_args,
        )
    )
