from typing import Any

from qdrant_odm.model.base import QdrantModel
from qdrant_odm.model.metadata import ModelMetadata


def get_model_metadata(model: type[QdrantModel]) -> ModelMetadata:
    """
    Return the compiled ODM metadata for a model class.

    Args:
        model:
            The ODM model class to inspect.

    Returns:
        The model metadata collected during subclass initialization.
    """
    return model.schema_definition()


def collection_or_raise(model: type[QdrantModel]) -> str:
    """
    Return the declared collection name for a model class.

    Args:
        model:
            The ODM model class to inspect.

    Returns:
        The collection name as a string.

    Raises:
        ValueError:
            If the model does not define `__collection__`.
    """
    name = getattr(model, "__collection__", "")
    if not name:
        raise ValueError(f"{model.__name__} must define __collection__")
    return str(name)


def safe_model_dump(model: QdrantModel, *, exclude: set[str]) -> dict[str, Any]:
    """
    Serialize a model instance using ODM-safe defaults.

    This helper dumps the model in Python mode, applies field aliases,
    and excludes the requested fields.

    Args:
        model:
            The model instance to serialize.
        exclude:
            A set of field names to exclude from the serialized output.

    Returns:
        A serialized payload dictionary.
    """
    return model.model_dump(mode="python", by_alias=True, exclude=exclude)