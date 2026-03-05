"""Decorator for wrapping __init__ methods of classes that needs to be linked to a registry."""

from functools import wraps
from typing import Any, Callable

from nml.state.system.registry import InstanceRegistry

__all__ = ["LinkToRegistry"]


class LinkToRegistry:
    """Generates a decorator for __init__ that links the construction to a given registry."""

    registry: InstanceRegistry

    def __init__(self, registry: None | InstanceRegistry = None) -> None:
        """Create new registry if not given."""
        self.registry = registry if registry is not None else InstanceRegistry()

    def __call__[T](self, constructor: Callable[..., T]) -> Callable[..., T]:
        """Run init as is, register new instance."""

        @wraps(constructor)
        def new_constructor(
            cls: type[T],
            *args: Any,
            _id: None | int = None,
            **kwargs: Any,
        ) -> T:
            """Run init as is, register new instance."""
            # If id is provided and object exists, skip construction (useful in reduce)
            if _id is not None and _id in self.registry:
                return self.registry[_id]  # type: ignore[return-value]

            # Construct object for the first time
            obj = constructor(cls)
            # Register object (provide id so that if _id is provided then that is
            # used to name the object, this is useful when unpickling)
            self.registry.register(obj, obj_id=_id)

            return obj

        return new_constructor
