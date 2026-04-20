from typing import Any

from qdrant_odm.model.base import QdrantModel
from qdrant_odm.model.metadata import ModelMetadata


def get_model_metadata(model: type[QdrantModel]) -> ModelMetadata:
    return model.schema_definition()


def collection_or_raise(model: type[QdrantModel]) -> str:
    name = getattr(model, "__collection__", "")
    if not name:
        raise ValueError(f"{model.__name__} must define __collection__")
    return str(name)


def safe_model_dump(model: QdrantModel, *, exclude: set[str]) -> dict[str, Any]:
    return model.model_dump(mode="python", by_alias=True, exclude=exclude)
