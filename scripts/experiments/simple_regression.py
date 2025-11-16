"""Simple regression."""

from pathlib import Path

from mlflow.client import MlflowClient
from nml.callbacks.compose import CallbackCat
from nml.callbacks.logging.mlflow import MLFlowMetricCallback
from nml.callbacks.ui.progress import ProgressCallback
from nml.itf.torch.train.trainer.base import TorchTrainer
from nml.itf.torch.train.trainer.components.batch_loss.base import TorchBatchLossTrainer
from nml.itf.torch.train.trainer.components.optimizer_map.standard import (
    get_standard_optimizer_map,
)
from nml.itf.torch.utils.loss.aggregators.mean import TorchMeanAggregator
from nml.loss.runner.standard.ml import MLLossRunner
from nml.tools.torch.data.dataset.linear import generate_regression
from nml.tools.torch.data.loader.container.array import ArrayDataLoader
from nml.tools.torch.models.modules.ff import ff_regression_model
from nml.utils.mlflow.client import ExperimentClient
from torch import nn
from torch.optim import Adam

if __name__ == "__main__":
    # Set parameters
    num_epochs = 10
    num_inputs = 5
    num_targets = 2
    hidden_layers = [100, 50]
    n_samples = 200

    layers = [num_inputs] + hidden_layers + [num_targets]

    lr = 1e-5

    mlflow_uri = Path("results/mlflow").absolute().as_uri()

    # Create model
    model = ff_regression_model(layers=layers, activations=nn.ReLU())

    # Cerate dataset
    # TODO: add splitting functionality
    dataset = ArrayDataLoader.from_torch_dataset(
        name="train",
        dataset=generate_regression(
            n_samples=n_samples, n_features=num_inputs, n_targets=num_targets
        ),
    )

    # Create trainer
    trainer = TorchTrainer(
        num_epochs=num_epochs,
        batch_loss_trainer=TorchBatchLossTrainer(
            optimizer_map=get_standard_optimizer_map(Adam, lr=lr)
        ),
        loss_aggregator=TorchMeanAggregator(),
    )

    # Create callbacks
    callback = CallbackCat(
        callbacks=(
            ProgressCallback(),
            MLFlowMetricCallback(
                client=ExperimentClient(
                    client=MlflowClient(tracking_uri=mlflow_uri),
                    in_experiment_name="test",
                )
            ),
        )
    )

    # Launch training
    trainer.train(
        model=model,
        loss_runner=MLLossRunner(),
        dataset_splits=(dataset,),
        callback=callback,
    )
