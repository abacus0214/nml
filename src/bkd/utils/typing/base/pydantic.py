"""Basic pydantic helper types."""

from pydantic import BaseModel, ConfigDict

__all__ = ["RestrictedBaseModel", "StandardBaseModel"]


class StandardBaseModel(BaseModel):
    """Base model that does not allow for extra inputs."""

    model_config = ConfigDict(
        extra="forbid",
    )


class RestrictedBaseModel(BaseModel):
    """Base model that does not allow for extra inputs."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )
