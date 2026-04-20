"""Shared package-level types and default constants."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from qdrant_odm.model.base import QdrantModel

T = TypeVar("T", bound=QdrantModel)


@dataclass(slots=True)
class ScrollPage(Generic[T]):
    """
    One page of scroll results together with the offset for the next request.

    Attributes:
        items:
            The deserialized model instances returned in the current page.
        next_offset:
            The offset token to use for the next scroll request, or None if there
            are no more results.
    """

    items: list[T]
    next_offset: Any | None = None


DEFAULT_UPSERT_BATCH_SIZE = 100
DEFAULT_RETRIEVE_CHUNK_SIZE = 128