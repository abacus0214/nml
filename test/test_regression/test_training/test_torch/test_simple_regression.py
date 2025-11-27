"""Simple regression."""

from typing import Any

from _pytest.tmpdir import TempPathFactory
from mlflow.client import MlflowClient
from nml.callbacks.abc import EventCallback
from nml.callbacks.compose import CallbackCat
from nml.callbacks.logging.mlflow import MLFlowMetricCallback
from nml.itf.torch.models.container.base import TorchModel
from nml.itf.torch.train.trainer.base import TorchTrainer
from nml.itf.torch.train.trainer.components.batch_loss.base import TorchBatchLossTrainer
from nml.itf.torch.train.trainer.components.evaluator.metrics import TorchEvalEvaluator
from nml.itf.torch.train.trainer.components.optimizer_map.standard import (
    get_standard_optimizer_map,
)
from nml.itf.torch.utils.loss.aggregators.mean import TorchMeanAggregator
from nml.loss.runner.standard.ml import MLLossRunner
from nml.tools.torch.data.dataset.linear import generate_regression
from nml.tools.torch.data.loader.container.tensor import TensorLoaderContainer
from nml.tools.torch.models.interpreter.normal import NormalOutType
from nml.tools.torch.models.modules.ff import ff_regression_model
from nml.tools.train.trainer.components.evaluator.plots.base import PlotlyEvaluator
from nml.tools.train.trainer.components.evaluator.plots.dataset import (
    DatasetPlotGenerator,
)
from nml.tools.train.trainer.components.evaluator.plots.regression import (
    RegressionPlotGenerator,
)
from nml.tools.utils.frequencer.standard import AtEnd, AtStart, Every, For
from nml.train.trainer.components.evaluator.base import EvaluatorABC
from nml.utils.mlflow.client import ExperimentClient, RunClient
from pytest import fixture, mark
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import random_split
from torcheval.metrics.regression import R2Score


@fixture(scope="class")
def mlflow_uri(tmp_path_factory: TempPathFactory) -> str:
    """Write mlflow in temp directory."""
    return tmp_path_factory.mktemp("mlflow").absolute().as_uri()


@fixture(scope="class")
def layers(num_inputs: int, hidden_layers: list[int], num_targets: int) -> list[int]:
    """Get number of layers for model."""
    return [num_inputs] + hidden_layers + [num_targets]


@fixture(scope="class")
def model(layers: list[int]) -> TorchModel[Tensor, NormalOutType, Tensor]:
    """Create torch model to be trained."""
    return ff_regression_model(layers=layers, activations=nn.ReLU())


@fixture(scope="class")
def dataset_loaders(
    n_samples: int, num_inputs: int, num_targets: int, batch_size: int
) -> tuple[TensorLoaderContainer, TensorLoaderContainer]:
    """Create trian and val data loaders."""
    # Cerate dataset
    dataset = generate_regression(
        n_samples=n_samples,
        n_features=num_inputs,
        n_targets=num_targets,
    )

    # Split in val and test
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

    # Create loaders
    train_loader = TensorLoaderContainer.from_torch_dataset(
        dataset=train_dataset,  # type: ignore
        batch_size=batch_size,
    )
    val_loader = TensorLoaderContainer.from_torch_dataset(
        dataset=val_dataset,  # type: ignore
        batch_size=batch_size,
    )
    return train_loader, val_loader


@fixture(scope="class")
def trainer(num_epochs: int, lr: float) -> TorchTrainer:
    """Create torch trainer."""
    return TorchTrainer(
        num_epochs=num_epochs,
        batch_loss_trainer=TorchBatchLossTrainer(
            optimizer_map=get_standard_optimizer_map(Adam, lr=lr)
        ),
        loss_runner=MLLossRunner(),
        loss_aggregator=TorchMeanAggregator(),
    )


@fixture(scope="class")
def callbacks(mlflow_uri: str) -> tuple[EventCallback, MLFlowMetricCallback]:
    """Generate callback."""
    mlflow_metric_callback = MLFlowMetricCallback(
        client=ExperimentClient(
            client=MlflowClient(tracking_uri=mlflow_uri),
            in_experiment_name="test",
        )
    )
    return CallbackCat(callbacks=(mlflow_metric_callback,)), mlflow_metric_callback


@fixture(scope="class")
def evaluators() -> tuple[EvaluatorABC[Any], ...]:
    """Generate evaluators."""
    plot_evaluator = PlotlyEvaluator(
        plots={
            "data": DatasetPlotGenerator(),
            "regression": RegressionPlotGenerator(),
        },
        frequency={
            "data": AtStart() & For(dids={"train"}),
            "regression": (Every(freq=300) | AtEnd()) & For(dids={"val"}),
        },
    )
    torcheval_evaluator = TorchEvalEvaluator(
        metrics={"r2": R2Score()},
        frequency={"r2": Every(freq=5)},
    )

    return plot_evaluator, torcheval_evaluator


@fixture(scope="class")
def training_run(
    trainer: TorchTrainer,
    model: TorchModel[Tensor, NormalOutType, Tensor],
    dataset_loaders: tuple[TensorLoaderContainer, TensorLoaderContainer],
    evaluators: tuple[EvaluatorABC[Any], ...],
    callbacks: tuple[EventCallback, MLFlowMetricCallback],
) -> RunClient:
    """Train model."""
    # Separate datasets
    train_loader, val_loader = dataset_loaders

    trainer.train(
        model=model,
        train_splits={"train": train_loader},
        val_splits={"val": val_loader},
        evaluators=evaluators,
        callback=callbacks[0],
    )

    # Extract run
    current_run = callbacks[1].current_run

    if current_run is None:
        raise RuntimeError("Could not find run")
    return current_run


@mark.parametrize(
    [
        "num_epochs",
        "num_inputs",
        "num_targets",
        "hidden_layers",
        "n_samples",
        "batch_size",
        "lr",
        "expected_r2",
    ],
    [
        (
            500,
            1,
            1,
            [500, 100, 50],
            2000,
            128,
            1e-5,
            0.85,
        ),
    ],
    ids=["big_model_500_ep"],
    scope="class",
)
class TestTrainingResult:
    """Test several parts of the result of training."""

    def test_r2_score(
        self,
        training_run: RunClient,
        expected_r2: float,
    ) -> None:
        """Check that r2 score is as expected."""
        # Get r2 score
        r2_score = max(
            training_run.client.get_metric_history(training_run.run_id, "val/r2"),
            key=lambda m: m.step,
        )

        # Check value
        assert r2_score.value >= expected_r2
