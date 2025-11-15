"""Test checking a simple regression."""

from datetime import timedelta
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from nml.data.batches.interpreter.standard import TupleBatchInterpreter
from nml.itf.torch.data.loader.container.base import TorchDataLoader
from nml.itf.torch.models.container.base import TorchModel
from nml.itf.torch.train.trainer.base import TorchTrainer
from nml.itf.torch.train.trainer.components.batch_loss.base import TorchBatchLossTrainer
from nml.itf.torch.train.trainer.components.optimizer_map.standard import (
    get_standard_optimizer_map,
)
from nml.itf.torch.utils.loss.aggregators.mean import TorchMeanAggregator
from nml.loss.runner.standard.ml import MLLossRunner
from nml.tools.torch.data.dataset.linear import generate_regression
from nml.tools.torch.models.interpreter.normal import (
    NormalOutType,
    TorchNormalInterpeter,
)
from pytest import fixture
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader


@fixture(scope="class")
def min_num_samples() -> int:
    """Minimum numnber of datapoints."""
    return 50


@fixture(scope="class")
def max_num_samples() -> int:
    """Minimum numnber of datapoints."""
    return 150


@fixture(scope="class")
def lr() -> float:
    """Learning rate to be used."""
    return 1e-5


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
def trainer(num_epochs: int, lr: float) -> TorchTrainer:
    """Create torch trainer."""
    return TorchTrainer(
        num_epochs=num_epochs,
        batch_loss_trainer=TorchBatchLossTrainer(
            optimizer_map=get_standard_optimizer_map(Adam, lr=lr)
        ),
        loss_aggregator=TorchMeanAggregator(),
    )


@st.composite
def torch_dataset(
    draw: st.DrawFn,
    num_inputs: int,
    min_num_samples: int,
    max_num_samples: int,
    num_out: int,
) -> TorchDataLoader[tuple[Any, Any], Any, Any]:
    """Fixture for torch dataset."""
    # Generate dataset
    dataset = generate_regression(
        n_samples=draw(
            st.integers(min_value=min_num_samples, max_value=max_num_samples)
        ),
        n_features=num_inputs,
        n_targets=num_out,
    )

    # Wrap inside loader
    return TorchDataLoader(
        name="test_data",
        torch_loader=DataLoader[Any](dataset),
        batch_interpreter=TupleBatchInterpreter[Any, Any](),
    )


@pytest.mark.parametrize(
    ["num_inputs", "num_hidden", "num_out"],
    [
        (5, 20, 1),
    ],
    ids=["toy"],
    scope="class",
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

    @given(data=st.data())
    def test_data_reading(
        self,
        data: st.DataObject,
        num_inputs: int,
        min_num_samples: int,
        max_num_samples: int,
        num_out: int,
        model: TorchModel[Any, Any, Any],
    ) -> None:
        """Try to run trainer."""
        # Generate dataset
        loader = data.draw(
            torch_dataset(
                num_inputs=num_inputs,
                min_num_samples=min_num_samples,
                max_num_samples=max_num_samples,
                num_out=num_out,
            )
        )

        # Loop through loader and use reader
        for batch in loader.epoch_iterator:
            ipt_manual = loader.batch_interpreter.get_ipt(batch.batch)
            tgt_manual = loader.batch_interpreter.get_tgt(batch.batch)

            ipt = batch.ipt
            tgt = batch.tgt

            assert (ipt == ipt_manual).all()
            assert (tgt == tgt_manual).all()

            assert isinstance(ipt, Tensor)
            assert isinstance(tgt, Tensor)
            assert ipt.shape[-1] == num_inputs
            assert tgt.shape[-1] == num_out

    @pytest.mark.parametrize("num_epochs", [10], ids=["short"], scope="class")
    @given(data=st.data())
    @settings(deadline=timedelta(minutes=5), max_examples=3)
    def test_training(
        self,
        data: st.DataObject,
        trainer: TorchTrainer,
        model: TorchModel[Any, Any, Any],
        num_inputs: int,
        min_num_samples: int,
        max_num_samples: int,
        num_out: int,
    ) -> None:
        """Try to run trainer."""
        # Generate dataset
        loader = data.draw(
            torch_dataset(
                num_inputs=num_inputs,
                min_num_samples=min_num_samples,
                max_num_samples=max_num_samples,
                num_out=num_out,
            )
        )

        # Training loop
        trainer.train(
            model=model,
            loss_runner=MLLossRunner(),
            dataset_splits=(loader,),
        )
