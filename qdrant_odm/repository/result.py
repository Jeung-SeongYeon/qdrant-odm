from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from qdrant_odm.model.base import QdrantModel

T = TypeVar("T", bound=QdrantModel)


class SearchHit(BaseModel, Generic[T]):
    """
    Typed search result wrapper for repository search operations.

    A search hit contains:
    - the Qdrant point id,
    - the final score,
    - the returned payload,
    - optional returned vector data,
    - the deserialized ODM model instance.

    Attributes:
        id:
            The Qdrant point id.
        score:
            The search score returned by Qdrant or a fused score in hybrid search.
        payload:
            The payload dictionary returned from Qdrant.
        vectors:
            Optional vector data returned from Qdrant when requested.
        document:
            The deserialized model instance reconstructed from the point.
    """

    id: Any
    score: float
    payload: dict[str, Any]
    vectors: dict[str, Any] | list[float] | None = None
    document: T

    model_config = {"arbitrary_types_allowed": True}