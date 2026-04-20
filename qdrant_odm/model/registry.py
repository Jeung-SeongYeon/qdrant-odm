from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_odm.model.base import QdrantModel

_REGISTRY: dict[str, type[QdrantModel]] = {}


def register_model(model: type[QdrantModel]) -> None:
    """
    Register a model class by its collection name.

    This allows later lookup from a collection name back to the model class.

    Args:
        model:
            The `QdrantModel` subclass to register.
    """
    _REGISTRY[model.collection_name()] = model


def get_model(collection_name: str) -> type[QdrantModel] | None:
    """
    Return the registered model for a collection name.

    Args:
        collection_name:
            The Qdrant collection name.

    Returns:
        The registered model class, or None if not found.
    """
    return _REGISTRY.get(collection_name)


def clear_registry() -> None:
    """
    Remove all registered models from the in-memory registry.
    """
    _REGISTRY.clear()