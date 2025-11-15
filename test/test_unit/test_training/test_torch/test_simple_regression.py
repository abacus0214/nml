"""Test checking a simple regression."""

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from nml.itf.torch.models.container.base import TorchModel
from nml.tools.torch.data.dataset.linear import generate_regression
from nml.tools.torch.models.interpreter.normal import (
    NormalOutType,
    TorchNormalInterpeter,
)
from pytest import fixture
from torch import Tensor, nn
from torch.utils.data import Dataset


@fixture(scope="class")
def layers(num_inputs: int, num_hidden: int, num_out: int) -> list[int]:
    """Generate the layer structure for the model."""
    return [num_inputs, num_hidden, num_out]


@fixture(scope="class")
def model(layers: list[int]) -> TorchModel[Tensor, NormalOutType, Tensor]:
    """Manually generate torch model for regression."""
    # Create container
    return TorchModel[Tensor, NormalOutType, Tensor](
        interpreter=TorchNormalInterpeter(),
        module=nn.Sequential(
            *(
                nn.Linear(in_ft, out_ft)
                for in_ft, out_ft in zip(layers[:-1], layers[1:])
            )
        ),
    )


@fixture(scope="class")
def dataset() -> Dataset:
    """Fixture for torch dataset."""
    return generate_regression()


@pytest.mark.parametrize(
    ["num_inputs", "num_hidden", "num_out"], [(10, 50, 1)], scope="class"
)
class TestSimpleRegressor:
    """Test suite for simple regression model."""

    def test_model_construction(self, model: TorchModel[Any, Any, Any]) -> None:
        """Try to run trainer."""
        # Check contents of the model
        assert model.interpreter is not None
        assert isinstance(model.module, nn.Module)

    @given(data=st.data())
    def test_model_log_likelihood(
        self,
        data: st.DataObject,
        num_inputs: int,
        num_out: int,
        model: TorchModel[Any, Any, Any],
    ) -> None:
        """Try to run trainer."""
        # Generate random input
        input_t = Tensor(data.draw(arrays(dtype=np.float32, shape=num_inputs)))
        target = Tensor(data.draw(arrays(dtype=np.float32, shape=num_out)))

        # Try to compute loss
        model.log_likelihood(ipt=input_t, target_samples=target)
