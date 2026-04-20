from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from qdrant_odm.exceptions import SchemaConflictError
from qdrant_odm.model.base import QdrantModel
from qdrant_odm.schema.qdrant_schema import INDEX_TYPES
from qdrant_odm.schema.diff import SchemaDiff, compute_schema_diff
from qdrant_odm.schema.planner import SchemaOperation, build_sync_operations


class SchemaManager:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client

    async def diff(self, model: type[QdrantModel]) -> SchemaDiff:
        return await compute_schema_diff(self.client, model)

    async def plan_sync(self, model: type[QdrantModel]) -> list[SchemaOperation]:
        schema_diff = await self.diff(model)
        self._raise_if_blocking(schema_diff)
        return build_sync_operations(model, schema_diff)

    async def dry_run(self, model: type[QdrantModel]) -> list[SchemaOperation]:
        return await self.plan_sync(model)

    async def sync(self, model: type[QdrantModel]) -> None:
        for operation in await self.plan_sync(model):
            if operation.operation == "create_collection":
                await self.client.create_collection(
                    collection_name=model.collection_name(),
                    vectors_config=self._build_vectors_config(model),
                    sparse_vectors_config=self._build_sparse_vectors_config(model),
                )
            if operation.operation == "create_payload_index":
                field_name = operation.detail
                payload_info = model.schema_definition().payload_fields[field_name]
                index_type = INDEX_TYPES.get(payload_info.index or "")
                if index_type is None:
                    raise SchemaConflictError(f"Unsupported index type: {payload_info.index!r}")
                await self.client.create_payload_index(
                    collection_name=model.collection_name(),
                    field_name=field_name,
                    field_schema=index_type,
                )

    def _raise_if_blocking(self, schema_diff: SchemaDiff) -> None:
        if schema_diff.vector_missing:
            raise SchemaConflictError(
                "Missing vectors in live collection: " + ", ".join(repr(name) for name in schema_diff.vector_missing)
            )
        if schema_diff.vector_mismatches:
            raise SchemaConflictError("; ".join(schema_diff.vector_mismatches))
        if schema_diff.sparse_missing:
            raise SchemaConflictError(
                "Missing sparse vectors in live collection: "
                + ", ".join(repr(name) for name in schema_diff.sparse_missing)
            )
        if schema_diff.payload_index_type_mismatches:
            raise SchemaConflictError("; ".join(schema_diff.payload_index_type_mismatches))

    def _build_vectors_config(self, model: type[QdrantModel]) -> dict[str, models.VectorParams]:
        meta = model.schema_definition()
        vectors_config: dict[str, models.VectorParams] = {}
        for vector_info in meta.vector_fields.values():
            vectors_config[vector_info.name] = models.VectorParams(
                size=vector_info.size,
                distance=vector_info.distance,
                on_disk=vector_info.on_disk,
            )
        return vectors_config

    def _build_sparse_vectors_config(self, model: type[QdrantModel]) -> dict[str, models.SparseVectorParams]:
        meta = model.schema_definition()
        sparse_vectors_config: dict[str, models.SparseVectorParams] = {}
        for sparse_info in meta.sparse_vector_fields.values():
            sparse_vectors_config[sparse_info.name] = models.SparseVectorParams()
        return sparse_vectors_config
