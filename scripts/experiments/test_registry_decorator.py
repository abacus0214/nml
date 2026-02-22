import pickle
from collections import UserDict
from functools import partial, wraps
from typing import Any, Callable
from inspect import getargs, Arguments

REGISTRY_ATTR_NAME = "__instances"


class InstanceRegistry(UserDict[int, object]):
    """Register instances of a particular class."""

    def register(self, obj: object, obj_id: None | int = None) -> None:
        """Register a new object."""
        # If id is not provided, generate one
        if obj_id is None:
            obj_id = int(id(obj))
        # Link object to id
        self[obj_id] = obj

    def get_id(self, obj: object) -> None | int:
        """Get the id of an object."""
        for obj_id, obj_t in self.items():
            if obj_t is obj:
                return obj_id

        return None

    @staticmethod
    def get_registry(obj: object) -> "InstanceRegistry":
        """Extract registry from obejct."""
        registry = getattr(obj, REGISTRY_ATTR_NAME)

        if not isinstance(registry, InstanceRegistry):
            raise TypeError(
                f"Object has attribute {REGISTRY_ATTR_NAME} but it is of type {type(registry)} not InstanceRegistry"
            )

        return registry


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
                return self.registry[_id]  # type: ingore

            # Construct object for the first time
            obj = constructor(cls)
            # Register object (provide id so that if _id is provided then that is
            # used to name the object, this is useful when unpickling)
            self.registry.register(obj, obj_id=_id)

            return obj

        return new_constructor


type ReduceOutType = (
    tuple[type[Any], tuple[Any, ...], dict[str, Any]]
    | tuple[type[Any], tuple[Any, ...]]
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
        ) -> None:
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
            _, (cls, _, _), kwargs = std_reduce_out

            # Recreate new positional arguments (skip first argument 'self')
            args = tuple((kwargs[arg_name] for arg_name in self.arguments.args[1:]))

            return (partial(cls, _id=self.registry.get_id(obj)), args, {})

        return new_reduce


class RegistryMeta(type):
    """Make classes a registry."""

    __instances__: InstanceRegistry

    def __init__(
        self,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Make the decorated class a registry."""
        super().__init__(name, bases, namespace, **kwargs)

        # Add registry field to the class
        self.__instances__ = InstanceRegistry()

        # Make it so init registers new instances
        self.__new__ = LinkToRegistry(self.__instances__)(self.__new__)
        self.__reduce__ = ReduceFromRegistry(
            registry=self.__instances__,
            arguments=getargs(self.__init__.__code__),
        )(self.__reduce__)

        # Cerate wrapper for init to ignore _id argument used by registry decorators
        orig_init = self.__init__

        @wraps(self.__init__)
        def ignore_id_init(*args: Any, _id: None | int = None, **kwargs: Any) -> None:
            """Wrap init function to ignore _id argument."""
            orig_init(*args, **kwargs)

        self.__init__ = ignore_id_init


class Test(metaclass=RegistryMeta):
    """Test class."""

    x: int
    y: float

    def __init__(self, x: int, y: float = 0.2) -> None:
        """Initialize."""
        self.x = x
        self.y = y


class Wrapper:
    t: Test

    def __init__(self, t: Test) -> None:
        self.t = t


NUM_OBJ = 3

# t = Test(x=1)
# t2 = Test(x=2)
# objects = [Wrapper(t=t) for _ in range(NUM_OBJ)]

# with open(f"test.pkl", "wb") as f:
#    pickle.dump(objects, f)
#
# with open(f"test.pkl", "rb") as f:
#    objects_loaded = pickle.load(f)

# for i in range(NUM_OBJ):
#    with open(f"test_{i}.pkl", "wb") as f:
#        pickle.dump(objects[i], f)

print("-------loading---------")
objects_loaded = []
for i in range(NUM_OBJ):
    with open(f"test_{i}.pkl", "rb") as f:
        objects_loaded.append(pickle.load(f))


def check(objects):
    assert all((x.t is objects[0].t for x in objects))


# check(objects)
check(objects_loaded)
