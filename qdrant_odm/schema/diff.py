from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient

from qdrant_odm.model.base import QdrantModel
from qdrant_odm.schema.qdrant_schema import (
    SUPPORTED_ODM_INDEX_TAGS,
    live_payload_index_tags,
    normalize_named_vectors,
    normalize_sparse_vectors,
)


@dataclass(slots=True)
class SchemaDiff:
    """
    Structured comparison between a model definition and a live Qdrant collection.

    This object captures the differences detected during schema inspection, including:
    - whether the collection exists,
    - missing or mismatched dense vectors,
    - missing sparse vectors,
    - missing payload indexes,
    - payload index type mismatches,
    - extra vector names present only in the live collection.
    """

    collection_exists: bool
    vector_missing: list[str] = field(default_factory=list)
    vector_mismatches: list[str] = field(default_factory=list)
    sparse_missing: list[str] = field(default_factory=list)
    payload_index_missing: list[str] = field(default_factory=list)
    payload_index_type_mismatches: list[str] = field(default_factory=list)
    extra_vector_names_in_collection: list[str] = field(default_factory=list)
    extra_sparse_vector_names_in_collection: list[str] = field(default_factory=list)

    def has_blocking_issues(self) -> bool:
        """
        Return whether the diff contains blocking issues that prevent automatic sync.

        Blocking issues include:
        - missing dense vectors,
        - mismatched dense vector definitions,
        - missing sparse vectors,
        - payload index type mismatches.

        Returns:
            True if the diff contains blocking issues, otherwise False.
        """
        return bool(
            self.vector_missing
            or self.vector_mismatches
            or self.sparse_missing
            or self.payload_index_type_mismatches
        )


async def compute_schema_diff(client: AsyncQdrantClient, model: type[QdrantModel]) -> SchemaDiff:
    """
    Compare a model schema definition against the live Qdrant collection state.

    If the collection does not exist, the diff reports the collection as missing and
    includes all declared indexed payload fields as missing payload indexes.

    If the collection exists, this function compares:
    - dense named vectors,
    - sparse named vectors,
    - payload index definitions.

    Args:
        client:
            The asynchronous Qdrant client instance.
        model:
            The ODM model class to compare.

    Returns:
        A structured schema diff describing the gap between the desired model schema
        and the live collection configuration.
    """
    meta = model.schema_definition()
    exists = await client.collection_exists(meta.collection_name)
    if not exists:
        missing_indexes = [
            field_name
            for field_name, payload_info in meta.payload_fields.items()
            if payload_info.index is not None
        ]
        return SchemaDiff(
            collection_exists=False,
            payload_index_missing=missing_indexes,
        )

    collection = await client.get_collection(meta.collection_name)
    params = collection.config.params
    live_vectors = normalize_named_vectors(params.vectors)
    live_sparse = normalize_sparse_vectors(params.sparse_vectors)
    live_payload_tags = live_payload_index_tags(collection.payload_schema or {})

    desired_vector_names = {info.name for info in meta.vector_fields.values()}
    desired_sparse_names = {info.name for info in meta.sparse_vector_fields.values()}

    vector_missing: list[str] = []
    vector_mismatches: list[str] = []
    for vector_info in meta.vector_fields.values():
        live = live_vectors.get(vector_info.name)
        if live is None:
            vector_missing.append(vector_info.name)
            continue
        if int(live.size) != int(vector_info.size):
            vector_mismatches.append(
                f"Vector size mismatch for {vector_info.name!r}: current={live.size}, desired={vector_info.size}"
            )
        if str(live.distance) != str(vector_info.distance):
            vector_mismatches.append(
                f"Vector distance mismatch for {vector_info.name!r}: current={live.distance}, desired={vector_info.distance}"
            )

    sparse_missing: list[str] = []
    for sparse_info in meta.sparse_vector_fields.values():
        if sparse_info.name not in live_sparse:
            sparse_missing.append(sparse_info.name)

    payload_index_missing: list[str] = []
    payload_index_type_mismatches: list[str] = []
    for field_name, payload_info in meta.payload_fields.items():
        if payload_info.index is None:
            continue
        desired_tag = payload_info.index
        if desired_tag not in SUPPORTED_ODM_INDEX_TAGS:
            payload_index_type_mismatches.append(
                f"Unsupported desired index type for field {field_name!r}: {desired_tag!r}"
            )
            continue
        if field_name not in live_payload_tags:
            payload_index_missing.append(field_name)
            continue
        live_tag = live_payload_tags[field_name]
        if live_tag != desired_tag:
            payload_index_type_mismatches.append(
                f"Payload index type mismatch for field {field_name!r}: live={live_tag!r}, desired={desired_tag!r}"
            )

    extra_vectors = sorted(name for name in live_vectors if name not in desired_vector_names and name != "")
    extra_sparse = sorted(name for name in live_sparse if name not in desired_sparse_names)

    return SchemaDiff(
        collection_exists=True,
        vector_missing=vector_missing,
        vector_mismatches=vector_mismatches,
        sparse_missing=sparse_missing,
        payload_index_missing=payload_index_missing,
        payload_index_type_mismatches=payload_index_type_mismatches,
        extra_vector_names_in_collection=extra_vectors,
        extra_sparse_vector_names_in_collection=extra_sparse,
    )