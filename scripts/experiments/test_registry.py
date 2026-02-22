from typing import Any, Callable

from abc import ABCMeta
from pydantic import BaseModel, Field, _internal
from typing import Callable, Protocol

SINGLETON_ATTR_NAME = "__singleton__"


class Constructor[T](Protocol):
    """Define the type of a constructor (__new__)."""

    def __call__(self, cls: type[T], *args: Any, **kwargs: Any) -> T:
        """General form of a constructor."""


class Init[T](Protocol):
    """Define the type of a constructor (__new__)."""

    def __call__(this, self: T, *args: Any, **kwargs: Any) -> None:
        """General form of an init method."""


class SingletonConstructor[T]:
    """Callable to replace the constructor for a new singleton instnace."""

    orig_constructor: Constructor[T]
    orig_init: Init[T]
    singleton: None | T = None

    def __init__(self, orig_constructor: Constructor[T], orig_init: Init[T]) -> None:
        """Store original constructor."""
        self.orig_constructor = orig_constructor
        self.orig_init = orig_init

    def __call__(self, cls: type[T], *args: Any, **kwargs: Any) -> T:
        """Logic to make T a singleton."""
        # If this is the first time instanciating singleton, generate the object normally
        if self.singleton is None:
            self.singleton = self.orig_constructor(cls)
            self.orig_init(self.singleton, *args, **kwargs)

        return self.singleton


def empty_init(*args: Any, **kwargs: Any) -> None:
    """Do nothing init."""


class SingletonMeta(type):
    """Metaclass to create singletons."""

    def __init__(
        self,
        name: str,
        bases: tuple[type[Any], ...],
        attr: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Make the class a singleton."""
        # Make class constructor a singleton
        self.__new__ = SingletonConstructor(
            orig_constructor=self.__new__, orig_init=self.__init__
        )
        self.__init__ = empty_init


counter = {"x": 0}


class TestSingle(metaclass=SingletonMeta):
    """Test singleton."""

    x: int
    y: float

    def __init__(self, x: int = 2, y: float = 3) -> None:
        counter["x"] = counter["x"] + 1
        self.x = x
        self.y = y


l = [TestSingle(x=i) for i in range(10)]

assert counter["x"] == 1, f"Counter is {counter['x']}"
assert all((x is l[0] for x in l))

breakpoint()


class RegisteredType(_internal._model_construction.ModelMetaclass):
    """Metaclass for objects that need to adhere to a registry pattern."""

    def __new__(
        cls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        __pydantic_generic_metadata__: _internal._generics.PydanticGenericMetadata
        | None = None,
        __pydantic_reset_parent_namespace__: bool = True,
        _create_model_module: str | None = None,
        **kwargs: Any,
    ) -> type:
        """Set registry for subclass."""
        # Build class as usual
        out_cls = super().__new__(
            cls,
            cls_name=cls_name,
            bases=bases,
            namespace=namespace,
            __pydantic_generic_metadata__=__pydantic_generic_metadata__,
            __pydantic_reset_parent_namespace__=__pydantic_reset_parent_namespace__,
            _create_model_module=_create_model_module,
            **kwargs,
        )

        return out_cls


class RegisteredTypeSimple(ABCMeta):
    """Metaclass for objects that need to adhere to a registry pattern."""

    def __new__(
        cls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        """Set registry for subclass."""
        # Build class as usual
        out_cls = super().__new__(
            cls,
            name,
            bases,
            namespace,
            **kwargs,
        )

        return out_cls


class ModelTest(BaseModel, metaclass=RegisteredType):
    """Test."""

    x: int
    y: float = Field(default=0.2)


class NormalTest(metaclass=RegisteredTypeSimple):
    """Test."""

    x: int
    y: float

    def __init__(self, x: int, y: float = 0.2) -> None:
        """Save vals."""
        self.x = x
        self.y = y


tm = ModelTest(x=1)
tn = NormalTest(x=1)
breakpoint()
