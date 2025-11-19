"""Base class for Model Interpreter interfaces."""

from enum import StrEnum, auto

__all__ = ["ModelInterpreter", "PredictionType"]


class PredictionType(StrEnum):
    """Type of method being used over model output."""

    OUTPUT = auto()
    SAMPLE = auto()
    ML = auto()


class ModelInterpreter[ModelOutT, SampleT, LikelihoodT]:
    """A class which collects methods to help interpret/use the output of a model.

    The basic concept around a model intepreter is that the output of a machine
    learning model is essentially giving the parameters to a distribution. Operations
    such as computing the loss, sampling, getting a prediction, etc.. are just operations
    on this probability distribution. For example, in a standard linear regression example,
    the liear model is effectively predicting the mean of a Gaussian distribution (if MSE
    is used) with an arbitrary/fixed variance. Predicting is the same as the outout
    of the model because the mean of a Gaussian is the maximum likelihood sample.

    The idea of this class is similar to that of a wrapper, but the "wrapped" model
    is passed as an argument. This allows for a global model interpreter that can
    be reused for multiple models.

    To couple a model with its interpreter and have the more user coincise and
    common wrapper schema, one can simply have a class pairing a model with its
    interpreter (see ModelContainer).
    """

    def __call__(
        self, model_out: ModelOutT, tp: PredictionType = PredictionType.OUTPUT
    ) -> ModelOutT | SampleT:
        """Call method specified by `tp`.

        This method gives the flexibility of progammaticaly selecting a method,
        but looses clarity in the typehinting of the output. If this is not
        an issue, then it is advised to use `PredictionType` to record what
        method to use and forward to `__call__`
        """
        match tp:
            case PredictionType.OUTPUT:
                return self.output(model_out=model_out)
            case PredictionType.SAMPLE:
                return self.sample(model_out=model_out)
            case PredictionType.ML:
                return self.ml(model_out=model_out)
            case _:
                raise TypeError(f"Cannot parse type {tp}")

    def output(self, model_out: ModelOutT) -> ModelOutT:
        """Get the model prediction (for general purposes).

        By default it is just the model output. One should never interact with
        the model directly, but only via the interpreter, even if operation is as simple
        as an identity function. This because, as previously mentioned, it makes it easier
        to overwrite the behavior of accessing the output of a model if one wants to.

        Args:
            model_out: the raw output of the model

        Returns:
            the output of the model (row or processed if this applies)

        """
        # TODO: add method that assumes that input is detached as well, use this in metrics eval
        return model_out

    def inference(self, model_out: ModelOutT) -> ModelOutT:
        """Get the model prediction (for evaluation purposes).

        Same as `output` but special bheavior for inference can be added here (for instance
        for torch models here we can detach the tensor).

        Args:
            model_out: the raw output of the model

        Returns:
            the output of the model (row or processed if this applies)

        """
        # TODO: add method that assumes that input is detached as well, use this in metrics eval
        return self.output(model_out=model_out)

    def sample(self, model_out: ModelOutT) -> SampleT:
        """Get one sample based on the model output.

        This method should be stochastic and should sample from the distribution
        that the mdoel has given as output.

        Even though this is not an abstract method it should be implemented when
        defining a model interpreter. The default implementation allows for
        interpreters to only have a subsets of the methods implemented, which is
        fine as long as the user understands that this is the case.
        """
        raise NotImplementedError(f"{self} does not support sampling.")

    def ml(self, model_out: ModelOutT) -> SampleT:
        """Get the maximum likelihood sample.

        This method is what is commonly referred to as the model prediction, which
        most of the time is the maximum likelihood sample from the model's output
        distribution.

        Even though this is not an abstract method it should be implemented when
        defining a model interpreter. The default implementation allows for
        interpreters to only have a subsets of the methods implemented, which is
        fine as long as the user understands that this is the case.
        """
        raise NotImplementedError(f"{self} does not support sampling.")

    def nll(self, target_samples: SampleT, model_out: ModelOutT) -> LikelihoodT:
        """Compute the log likelihood of a given set of sample based on the model output.

        The additive inverse of this is what is often used in Maximum Likelihood optimimization
        methods (e.g. MSE, MAE) where the log likelihood of the data acoording to the distribution
        given by the model is maximized (so the loss to be minimized is the additive inverse).
        """
        raise NotImplementedError(
            f"{self} does not support log likelihood computation."
        )
