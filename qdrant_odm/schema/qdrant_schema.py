from typing import Any

from qdrant_client.http import models

from qdrant_odm.model.fields import PayloadFieldInfo

INDEX_TYPES: dict[str, Any] = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "float": models.PayloadSchemaType.FLOAT,
    "bool": models.PayloadSchemaType.BOOL,
    "geo": models.PayloadSchemaType.GEO,
    "datetime": models.PayloadSchemaType.DATETIME,
    "text": models.PayloadSchemaType.TEXT,
    "uuid": models.PayloadSchemaType.UUID,
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


def _to_tokenizer(tokenizer: str | None):
    if tokenizer is None:
        return None
    mapping = {
        "word": models.TokenizerType.WORD,
        "whitespace": models.TokenizerType.WHITESPACE,
        "prefix": models.TokenizerType.PREFIX,
        "multilingual": models.TokenizerType.MULTILINGUAL,
    }
    return mapping[tokenizer]


def build_payload_index_schema(payload_info: PayloadFieldInfo) -> Any:
    """
    Build a Qdrant payload index schema object from ODM payload field metadata.
    """
    if payload_info.index == "keyword":
        if payload_info.keyword is None:
            return models.PayloadSchemaType.KEYWORD
        return models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=payload_info.keyword.is_tenant,
            on_disk=payload_info.keyword.on_disk,
            # enable_hnsw=payload_info.keyword.enable_hnsw,
        )

    if payload_info.index == "integer":
        if payload_info.integer is None:
            return models.PayloadSchemaType.INTEGER
        return models.IntegerIndexParams(
            type=models.IntegerIndexType.INTEGER,
            lookup=payload_info.integer.lookup,
            range=payload_info.integer.range,
            is_principal=payload_info.integer.is_principal,
            on_disk=payload_info.integer.on_disk,
            # enable_hnsw=payload_info.integer.enable_hnsw,
        )

    if payload_info.index == "float":
        if payload_info.float_ is None:
            return models.PayloadSchemaType.FLOAT
        return models.FloatIndexParams(
            type=models.FloatIndexType.FLOAT,
            is_principal=payload_info.float_.is_principal,
            on_disk=payload_info.float_.on_disk,
            # enable_hnsw=payload_info.float_.enable_hnsw,
        )

    if payload_info.index == "bool":
        if payload_info.bool_ is None:
            return models.PayloadSchemaType.BOOL
        return models.BoolIndexParams(
            type=models.BoolIndexType.BOOL,
            on_disk=payload_info.bool_.on_disk,
            # enable_hnsw=payload_info.bool_.enable_hnsw,
        )

    if payload_info.index == "geo":
        if payload_info.geo is None:
            return models.PayloadSchemaType.GEO
        return models.GeoIndexParams(
            type=models.GeoIndexType.GEO,
            on_disk=payload_info.geo.on_disk,
            # enable_hnsw=payload_info.geo.enable_hnsw,
        )

    if payload_info.index == "datetime":
        if payload_info.datetime is None:
            return models.PayloadSchemaType.DATETIME
        return models.DatetimeIndexParams(
            type=models.DatetimeIndexType.DATETIME,
            is_principal=payload_info.datetime.is_principal,
            on_disk=payload_info.datetime.on_disk,
            # enable_hnsw=payload_info.datetime.enable_hnsw,
        )

    if payload_info.index == "text":
        if payload_info.text is None:
            return models.PayloadSchemaType.TEXT
        return models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=_to_tokenizer(payload_info.text.tokenizer),
            min_token_len=payload_info.text.min_token_len,
            max_token_len=payload_info.text.max_token_len,
            lowercase=payload_info.text.lowercase,
            # ascii_folding=payload_info.text.ascii_folding,
            phrase_matching=payload_info.text.phrase_matching,
            stopwords=payload_info.text.stopwords,
            on_disk=payload_info.text.on_disk,
            stemmer=payload_info.text.stemmer,
            # enable_hnsw=payload_info.text.enable_hnsw,
        )

    if payload_info.index == "uuid":
        if payload_info.uuid is None:
            return models.PayloadSchemaType.UUID
        return models.UuidIndexParams(
            type=models.UuidIndexType.UUID,
            is_tenant=payload_info.uuid.is_tenant,
            on_disk=payload_info.uuid.on_disk,
            # enable_hnsw=payload_info.uuid.enable_hnsw,
        )

    raise ValueError(f"Unsupported payload schema: {payload_info.index!r}")


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _model_dump_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=False)
    if hasattr(obj, "dict"):
        return obj.dict(exclude_none=False)
    return obj


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def _normalize_payload_schema_name(value: Any) -> str:
    if value is None:
        return "unknown"
    if hasattr(value, "value"):
        return str(value.value).lower()
    if hasattr(value, "type"):
        t = getattr(value, "type", None)
        if t is not None:
            return _normalize_payload_schema_name(t)
    return str(value).lower()


def normalize_payload_index_object(value: Any) -> dict[str, Any]:
    """
    Normalize a live or desired payload index object into a plain comparison dict.
    """
    if value is None:
        return {}

    if isinstance(value, dict):
        raw = dict(value)
    else:
        raw = _model_dump_safe(value)
        if not isinstance(raw, dict):
            return {"type": _normalize_payload_schema_name(value)}

    params = raw.get("params")
    if isinstance(params, dict):
        merged = {**raw, **params}
    else:
        merged = dict(raw)

    normalized = {}
    for key, val in merged.items():
        normalized[key] = _normalize_value(val)

    if "data_type" in normalized and "type" not in normalized:
        normalized["type"] = normalized["data_type"]

    if "type" in normalized:
        normalized["type"] = _normalize_payload_schema_name(normalized["type"])

    return normalized


def desired_payload_index_object(payload_info: PayloadFieldInfo) -> dict[str, Any]:
    """
    Normalize a desired ODM payload index into a comparison dict.
    """
    schema_obj = build_payload_index_schema(payload_info)
    normalized = normalize_payload_index_object(schema_obj)
    if "type" not in normalized and payload_info.index is not None:
        normalized["type"] = payload_info.index.lower()
    return normalized


def compare_payload_index(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """
    Compare only the keys explicitly set in the expected index object.
    """
    diffs: list[str] = []

    exp = {k: v for k, v in expected.items() if v is not None}
    act = {k: v for k, v in actual.items() if v is not None}

    for key, exp_value in exp.items():
        act_value = act.get(key)
        if act_value != exp_value:
            diffs.append(f"{key}: live={act_value!r}, desired={exp_value!r}")

    return diffs