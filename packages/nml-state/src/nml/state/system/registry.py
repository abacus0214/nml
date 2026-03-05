"""Object to keep track of instances."""

from collections import UserDict

__all__ = ["REGISTRY_ATTR_NAME", "InstanceRegistry"]

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
