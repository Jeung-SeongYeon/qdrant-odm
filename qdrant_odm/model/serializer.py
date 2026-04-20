from typing import Any, TypeVar

from qdrant_odm.model.base import QdrantModel

T = TypeVar("T", bound=QdrantModel)


def model_to_payload(model: QdrantModel) -> dict[str, Any]:
    return model.to_payload()


def model_from_point(cls: type[T], *, point_id: Any, payload: dict[str, Any] | None) -> T:
    return cls.from_point(point_id=point_id, payload=payload)
