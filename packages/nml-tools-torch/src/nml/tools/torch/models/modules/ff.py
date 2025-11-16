"""Simple FF module factories."""

from collections import UserDict

from nml.itf.torch.models.container.base import TorchModel
from nml.tools.torch.models.interpreter.normal import (
    NormalOutType,
    TorchNormalInterpeter,
)
from torch import Tensor, nn

__all__ = ["ff_regression_model", "ActivationsType", "ActivationsDict"]

type ActivationsType = None | nn.Module | dict[int, nn.Module]


class ActivationsDict(UserDict[int, nn.Module]):
    """Dict mapping layer index to its activation."""

    out_layer_id: int

    default_activation: nn.Module
    default_output_activation: bool = False

    def __init__(
        self,
        *,
        activations: ActivationsType = None,
        num_layers: None | int = None,
        default_output_activation: bool = False,
    ) -> None:
        """Build activation dict."""
        # Store total number of layers (useful to dientify output layer)
        self.out_layer_id = num_layers - 1 if num_layers is not None else -1
        self.default_output_activation = default_output_activation
        # Store default activation (if one is specified)
        if isinstance(activations, nn.Module):
            self.default_activation = activations
        else:
            self.default_activation = nn.Identity()

        # Fill in values (go through sanitization)
        if isinstance(activations, dict):
            for key, val in activations.items():
                self[key] = val

    def sanitize_key(self, key: int) -> int:
        """Remap output key to -1."""
        if key == self.out_layer_id:
            return -1
        if key > self.out_layer_id and self.out_layer_id > 0:
            raise ValueError(f"Invalid key {key}")
        return key

    def __getitem__(self, key: int) -> nn.Module:
        """Set -1 to be the key of the last layer."""
        return super().__getitem__(self.sanitize_key(key))

    def __setitem__(self, key: int, item: nn.Module) -> None:
        """Set -1 to be the key of the last layer."""
        return super().__setitem__(self.sanitize_key(key), item)

    def __missing__(self, key: int) -> nn.Module:
        """Set default activation value."""
        key = self.sanitize_key(key)
        if key > 0 or self.default_output_activation:
            return self.default_activation
        return nn.Identity()


def ff_regression_model(
    layers: list[int] | tuple[int],
    activations: ActivationsType = None,
) -> TorchModel[Tensor, NormalOutType, Tensor]:
    """Generate simple torch model with linear layers and regression output layer."""
    # Build dict for activations
    activations_dict = ActivationsDict(activations=activations, num_layers=len(layers))

    # Concatenate linear layers
    return TorchModel[Tensor, NormalOutType, Tensor](
        interpreter=TorchNormalInterpeter(),
        module=nn.Sequential(
            *(
                nn.Sequential(
                    nn.Linear(in_ft, out_ft),
                    activations_dict[layer_idx],
                )
                for (layer_idx, in_ft), out_ft in zip(
                    enumerate(layers[:-1]), layers[1:]
                )
            )
        ),
    )
