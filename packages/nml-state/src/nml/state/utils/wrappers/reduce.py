"""Classes for wrapping reudce in such a way that it will query the registry."""

from functools import partial, wraps
from inspect import Arguments
from typing import Any, Callable

from nml.state.system.registry import InstanceRegistry

__all__ = ["ReduceOutType", "ReduceFromRegistry"]

type ReduceOutType = (
    tuple[type[Any], tuple[Any, ...], dict[str, Any]]
    # | tuple[type[Any], tuple[Any, ...]]
)


class ReduceFromRegistry:
    """Generates a decorator for the reduce function that allows object recycling."""

    registry: InstanceRegistry
    arguments: None | Arguments = None

    def __init__(
        self,
        arguments: None | Arguments = None,
        registry: None | InstanceRegistry = None,
    ) -> None:
        """Create new registry if not given."""
        self.registry = registry if registry is not None else InstanceRegistry()
        self.arguments = arguments

    def __call__[T](
        self, reduce_fn: Callable[[T], ReduceOutType]
    ) -> Callable[[T], ReduceOutType]:
        """Run init as is, register new instance."""

        @wraps(reduce_fn)
        def new_reduce(
            obj: T,
        ) -> ReduceOutType:
            """Reduce function for registry types."""
            # TODO: handle automatic argument extraction
            if self.arguments is None:
                raise NotImplementedError(
                    "Currently not supporting automatic extraction of __init__ argument, they must be provided."
                )

            # TODO: handle kwargs only arguments
            if self.arguments.varargs is not None or self.arguments.varkw is not None:
                raise NotImplementedError("Currently not supporting varargs or varkw.")

            # Get standard reduce output
            std_reduce_out = reduce_fn(obj)

            # Check output type
            if len(std_reduce_out) > 3 or len(std_reduce_out) < 2:
                raise RuntimeError(f"Incorrect reduce output {std_reduce_out}")

            # TODO: currently this only holds for reconstructor schema
            # TODO fails if the class has no fields
            _, (cls, _, _), kwargs = std_reduce_out

            # Recreate new positional arguments (skip first argument 'self')
            args = tuple((kwargs[arg_name] for arg_name in self.arguments.args[1:]))

            # When rebuilding the object, pass the id argument so that it will link similar instances
            # TODO: here by editing the IDs in the same way one could make it so that unpickling
            # within the same runtime rebuilds the object the first time. Right now unpickling
            # within the same runtime will just do object lookup, which may not be expected.
            # TODO improve this to work with the reconstructor schema
            return (partial(cls, _id=self.registry.get_id(obj)), args, {})  # type: ignore[return-value]

        return new_reduce
