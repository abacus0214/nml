"""Basic pydantic helper types."""

from pydantic import BaseModel, ConfigDict

__all__ = ["RestrictedBaseModel", "StandardBaseModel"]


class StandardBaseModel(BaseModel):
    """Base model that does not allow for extra inputs."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class RestrictedBaseModel(BaseModel):
    """Base model that does not allow for extra inputs."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )
