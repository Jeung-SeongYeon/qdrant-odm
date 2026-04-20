from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from qdrant_odm.model.base import QdrantModel

T = TypeVar("T", bound=QdrantModel)


@dataclass(slots=True)
class ScrollPage(Generic[T]):
    """One page of scroll results plus the offset for the next request."""

    items: list[T]
    next_offset: Any | None = None


DEFAULT_UPSERT_BATCH_SIZE = 100
DEFAULT_RETRIEVE_CHUNK_SIZE = 128
