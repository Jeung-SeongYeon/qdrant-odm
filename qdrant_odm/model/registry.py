from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_odm.model.base import QdrantModel

_REGISTRY: dict[str, type[QdrantModel]] = {}


def register_model(model: type[QdrantModel]) -> None:
    _REGISTRY[model.collection_name()] = model


def get_model(collection_name: str) -> type[QdrantModel] | None:
    return _REGISTRY.get(collection_name)


def clear_registry() -> None:
    _REGISTRY.clear()
