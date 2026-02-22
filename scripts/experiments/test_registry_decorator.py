from typing import Any, Callable, Self
from functools import wraps
from collections import UserDict

REGISTRY_ATTR_NAME = "__instances__"


class InstanceRegistry(UserDict[int, object]):
    """Register instances of a particular class."""

    def register(self, obj: object) -> None:
        """Register a new object."""
        self[int(id(obj))] = obj

    @staticmethod
    def get_registry(obj: object) -> "InstanceRegistry":
        """Extract registry from obejct."""
        return getattr(obj, REGISTRY_ATTR_NAME)


class LinkToRegistry:
    """Generates a decorator for __init__ that links the construction to a given registry."""

    registry: InstanceRegistry

    def __init__(self, registry: None | InstanceRegistry = None) -> None:
        """Create new registry if not given."""
        self.registry = registry if registry is not None else InstanceRegistry()

    def __call__(self, init_fun: Callable[..., None]) -> Callable[..., None]:
        """Run init as is, register new instance."""

        @wraps(init_fun)
        def new_init(obj, *args: Any, **kwargs: Any) -> None:
            """Run init as is, register new instance."""
            init_fun(obj, *args, **kwargs)
            self.registry.register(obj)

        return new_init


def make_registry[T](cls: type[T]) -> type[T]:
    """Make the decorated class a registry."""
    # Create registry for the decorated class
    registry = InstanceRegistry()

    # Add registry field to the class
    setattr(cls, REGISTRY_ATTR_NAME, registry)
    # Make it so init registers new instances
    setattr(cls, "__init__", LinkToRegistry(registry)(getattr(cls, "__init__")))

    return cls


@make_registry
class Test:
    """Test class."""

    x: int
    y: float

    def __init__(self, x: int, y: float = 0.2) -> None:
        """Initialize."""
        self.x = x
        self.y = y


l = [Test(x=i) for i in range(10)]

print(l[0].__instances__)
print(l[0].__class__.__instances__)
print(l[0].__class__.__instances__ is l[0].__instances__)
instances = l[0].__instances__
