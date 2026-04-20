from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from qdrant_odm.model.base import QdrantModel

T = TypeVar("T", bound=QdrantModel)


class SearchHit(BaseModel, Generic[T]):
    id: Any
    score: float
    payload: dict[str, Any]
    vectors: dict[str, Any] | list[float] | None = None
    document: T

    model_config = {"arbitrary_types_allowed": True}
