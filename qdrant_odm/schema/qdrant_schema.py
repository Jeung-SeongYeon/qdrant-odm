from typing import Any

from qdrant_client.http import models

INDEX_TYPES: dict[str, Any] = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "float": models.PayloadSchemaType.FLOAT,
    "bool": models.PayloadSchemaType.BOOL,
    "datetime": models.PayloadSchemaType.DATETIME,
}

SUPPORTED_ODM_INDEX_TAGS = frozenset(INDEX_TYPES.keys())


def payload_schema_type_to_tag(data_type: models.PayloadSchemaType | str) -> str:
    """
    Normalize a Qdrant payload schema type into the ODM string tag format.

    This helper converts Qdrant enum values or raw string-like values into
    normalized lowercase string tags used internally by the ODM.

    Args:
        data_type:
            A Qdrant payload schema type or string-like representation.

    Returns:
        A normalized payload index tag string.
    """
    if isinstance(data_type, models.PayloadSchemaType):
        return str(data_type.value)
    value = getattr(data_type, "value", data_type)
    return str(value).lower()


def normalize_named_vectors(vectors: models.VectorsConfig | None) -> dict[str, models.VectorParams]:
    """
    Normalize a Qdrant dense vector config into a name-to-params mapping.

    Qdrant may expose vector configuration in different shapes depending on the
    collection definition. This helper normalizes those variants into a consistent
    dictionary form.

    Args:
        vectors:
            The raw Qdrant dense vector configuration.

    Returns:
        A dictionary mapping vector names to `VectorParams`.
    """
    if vectors is None:
        return {}
    if isinstance(vectors, dict):
        return dict(vectors)
    if isinstance(vectors, models.VectorParams):
        return {"": vectors}
    return {}


def normalize_sparse_vectors(
    sparse_vectors: models.SparseVectorsConfig | None,
) -> dict[str, models.SparseVectorParams]:
    """
    Normalize a Qdrant sparse vector config into a name-to-params mapping.

    Args:
        sparse_vectors:
            The raw Qdrant sparse vector configuration.

    Returns:
        A dictionary mapping sparse vector names to `SparseVectorParams`.
    """
    if sparse_vectors is None:
        return {}
    params = getattr(sparse_vectors, "params", sparse_vectors)
    if isinstance(params, dict):
        return dict(params)
    return {}


def live_payload_index_tags(payload_schema: dict[str, models.PayloadIndexInfo]) -> dict[str, str]:
    """
    Extract normalized payload index type tags from a live payload schema.

    Args:
        payload_schema:
            The live payload schema returned by Qdrant.

    Returns:
        A dictionary mapping field names to normalized payload index type tags.
    """
    result: dict[str, str] = {}
    for field_name, info in payload_schema.items():
        result[field_name] = payload_schema_type_to_tag(info.data_type)
    return result