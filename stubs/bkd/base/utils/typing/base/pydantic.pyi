from _typeshed import Incomplete
from pydantic import BaseModel

__all__ = ['RestrictedBaseModel', 'StandardBaseModel']

class StandardBaseModel(BaseModel):
    model_config: Incomplete

class RestrictedBaseModel(BaseModel):
    model_config: Incomplete
