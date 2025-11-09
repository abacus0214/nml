"""Base class for Model Interpreter interfaces."""

from typing import SupportsFloat

__all__ = ["ModelInterpreter"]


class ModelInterpreter[ModelOutT, SampleT, LikelihoodT]:
    """A class which collects methods to help interpret/use the output of a model."""

    def pred(self, model_out: ModelOutT) -> ModelOutT:
        """Get the model prediction, by default it is just the model output."""
        return model_out

    def sample(self, model_out: ModelOutT) -> SampleT:
        """Get one sample based on the model output."""
        raise NotImplementedError(f"{self} does not support sampling.")

    def log_likelihood(
        self, target_samples: SampleT, model_out: ModelOutT
    ) -> LikelihoodT:
        """Compute the log likelihood of a given set of sample based on the model output."""
        raise NotImplementedError(
            f"{self} does not support log likelihood computation."
        )

    def likelihood_to_float(self, likelihood: LikelihoodT) -> float:
        """Convert the likelihood to a float."""
        if isinstance(likelihood, SupportsFloat):
            return float(likelihood)

        raise RuntimeError(
            f"Could not cast {likelihood} to float with default strategy."
        )
