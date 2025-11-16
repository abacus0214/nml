"""Some routines to generate linear data."""

from nml.tools.torch.data.dataset.array import ArrayDataset
from numpy.random import RandomState
from sklearn.datasets import make_regression
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["generate_regression"]


def generate_regression(
    n_samples: int = 100,
    n_features: int = 100,
    *,
    n_informative: int = 10,
    n_targets: int = 1,
    bias: float = 0.0,
    effective_rank: None | int = None,
    tail_strength: float = 0.5,
    noise: float = 0.0,
    shuffle: bool = True,
    random_state: None | int | RandomState = None,
) -> Dataset[tuple[Tensor, Tensor]]:
    """Generate a linear regressiomn problem with sklearn and then wrap it in a torch Dataset."""
    # Generate the dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_targets=n_targets,
        bias=bias,
        effective_rank=effective_rank,
        tail_strength=tail_strength,
        noise=noise,
        shuffle=shuffle,
        coef=False,
        random_state=random_state,
    )

    # Generate torch.utils.data.Dataset object
    return ArrayDataset(X=X, y=y)
