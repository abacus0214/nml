"""Mlflow event callback to log torch mdoels."""

from typing import Any

from mlflow.pytorch import save_model
from nml.callbacks.abc import EventCallback
from nml.itf.torch.models.container.base import TorchModel
from nml.models.container.base import ModelABC
from nml.utils.typing.base.pydantic import RestrictedBaseModel
from nml.utils.typing.eval.metrics import MetricStep
from nml.utils.typing.models.container.base import ModelID


class MlflowTorchModelCallback(RestrictedBaseModel, EventCallback):
    """Callback that logs torch models to mlflow."""

    # TODO: also have pyfunc flavor

    def log_model(
        self,
        model: ModelABC[Any, Any, Any, Any],
        name: None | ModelID = None,
        step: MetricStep = None,
    ) -> None:
        """Log a model."""
        # Ignore models that cannot be logged
        if not isinstance(model, TorchModel):
            return

        # Log pytorch model
        # TODO: log entire ModelABC object to be able to run it?
        save_model(model.module, name)
