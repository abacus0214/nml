"""Meta class to make class retain instance consistency."""

from functools import wraps
from inspect import getargs
from typing import Any

from nml.state.system.registry import InstanceRegistry
from nml.state.utils.wrappers.constructor import LinkToRegistry
from nml.state.utils.wrappers.reduce import ReduceFromRegistry


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
        self.__new__ = LinkToRegistry(self.__instances__)(self.__new__)  # type: ignore[assignment]
        self.__reduce__ = ReduceFromRegistry(  # type: ignore[assignment]
            registry=self.__instances__,
            arguments=getargs(self.__init__.__code__),  # type: ignore[misc]
        )(self.__reduce__)  # type: ignore[arg-type]

        # Cerate wrapper for init to ignore _id argument used by registry decorators
        orig_init = self.__init__  # type: ignore[misc]

        # Do not pass _id argument to __init__
        @wraps(orig_init)
        def ignore_id_init(*args: Any, _id: None | int = None, **kwargs: Any) -> None:
            """Wrap init function to ignore _id argument."""
            orig_init(*args, **kwargs)

        self.__init__ = ignore_id_init  # type: ignore[misc]
