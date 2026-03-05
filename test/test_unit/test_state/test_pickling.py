"""Test the nml-state package."""

import pickle
from pathlib import Path

from _pytest.tmpdir import TempPathFactory
from nml.state.system.meta import RegistryMeta
from pytest import fixture


class Test(metaclass=RegistryMeta):
    """Test class."""

    def __init__(self, x: int = 1) -> None:
        """Do nothing."""
        self.x = x


@fixture(scope="package")
def obj() -> object:
    """Object to use in pickling."""
    return Test()


class Wrapper:
    """Simple class to store another object inside."""

    obj: object

    def __init__(self, obj: object) -> None:
        """Store object."""
        self.obj = obj


@fixture(scope="package")
def num_objects() -> int:
    """Return number of objects to create."""
    return 10


def test_instance_consistency(
    obj: object, tmp_path_factory: TempPathFactory, num_objects: int
) -> None:
    """Try to pickle an object multiple times, see if it remains the same instance."""
    # Create directory in which to store pickle files
    tmp_path = Path(tmp_path_factory.mktemp("pickles"))

    # Create multiple objects containing the same field
    objects = [Wrapper(obj=obj) for _ in range(num_objects)]

    for i in range(num_objects):
        with open(tmp_path / f"test_{i}.pkl", "wb") as f:
            pickle.dump(objects[i], f)

    objects_loaded = []
    for i in range(num_objects):
        with open(tmp_path / f"test_{i}.pkl", "rb") as f:
            objects_loaded.append(pickle.load(f))

    assert all((x.obj is objects[0].obj for x in objects_loaded))
