from typing import Any, TypeVar

from qdrant_odm.model.base import QdrantModel

T = TypeVar("T", bound=QdrantModel)


def model_to_payload(model: QdrantModel) -> dict[str, Any]:
    """
    Serialize a model instance into a Qdrant payload dictionary.

    Args:
        model:
            The model instance to serialize.

    Returns:
        A payload dictionary ready to be stored in Qdrant.
    """
    return model.to_payload()


def model_from_point(cls: type[T], *, point_id: Any, payload: dict[str, Any] | None) -> T:
    """
    Reconstruct a model instance from a Qdrant point id and payload.

    Args:
        cls:
            The target model class.
        point_id:
            The Qdrant point id.
        payload:
            The payload dictionary retrieved from Qdrant.

    Returns:
        A validated model instance of type `cls`.
    """
    return cls.from_point(point_id=point_id, payload=payload)