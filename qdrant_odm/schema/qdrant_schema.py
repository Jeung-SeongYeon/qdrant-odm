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
    if isinstance(data_type, models.PayloadSchemaType):
        return str(data_type.value)
    value = getattr(data_type, "value", data_type)
    return str(value).lower()


def normalize_named_vectors(vectors: models.VectorsConfig | None) -> dict[str, models.VectorParams]:
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
    if sparse_vectors is None:
        return {}
    params = getattr(sparse_vectors, "params", sparse_vectors)
    if isinstance(params, dict):
        return dict(params)
    return {}


def live_payload_index_tags(payload_schema: dict[str, models.PayloadIndexInfo]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field_name, info in payload_schema.items():
        result[field_name] = payload_schema_type_to_tag(info.data_type)
    return result
