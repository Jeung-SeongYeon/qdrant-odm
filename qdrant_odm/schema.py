from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from qdrant_odm.exceptions import SchemaConflictError
from qdrant_odm.model import QdrantModel


INDEX_TYPES: dict[str, Any] = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "float": models.PayloadSchemaType.FLOAT,
    "bool": models.PayloadSchemaType.BOOL,
    "datetime": models.PayloadSchemaType.DATETIME,
}


@dataclass(slots=True)
class SchemaOperation:
    operation: str
    detail: str


class SchemaManager:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client

    async def sync(self, model: type[QdrantModel]) -> None:
        plan = await self.plan_sync(model)
        for operation in plan:
            if operation.operation == "create_collection":
                await self.client.create_collection(
                    collection_name=model.collection_name(),
                    vectors_config=self._build_vectors_config(model),
                    sparse_vectors_config=self._build_sparse_vectors_config(model),
                )
            if operation.operation == "create_payload_index":
                field_name = operation.detail
                payload_info = model.schema_definition().payload_fields[field_name]
                index_type = INDEX_TYPES.get(payload_info.index)
                if index_type is None:
                    raise SchemaConflictError(f"Unsupported index type: {payload_info.index!r}")
                await self.client.create_payload_index(
                    collection_name=model.collection_name(),
                    field_name=field_name,
                    field_schema=index_type,
                )

    async def plan_sync(self, model: type[QdrantModel]) -> list[SchemaOperation]:
        meta = model.schema_definition()
        exists = await self.client.collection_exists(meta.collection_name)
        operations: list[SchemaOperation] = []

        if not exists:
            operations.append(SchemaOperation(operation="create_collection", detail=meta.collection_name))
        else:
            await self._validate_live_vectors(model)
            await self._validate_live_sparse_vectors(model)

        operations.extend(self._collect_payload_index_operations(model))
        return operations

    async def dry_run(self, model: type[QdrantModel]) -> list[SchemaOperation]:
        return await self.plan_sync(model)

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

    async def _validate_live_vectors(self, model: type[QdrantModel]) -> None:
        meta = model.schema_definition()
        collection = await self.client.get_collection(meta.collection_name)
        config = collection.config.params
        live_vectors = getattr(config.vectors, "params", config.vectors)
        live_vectors_map = live_vectors if isinstance(live_vectors, dict) else {}

        for vector_info in meta.vector_fields.values():
            live_vector = live_vectors_map.get(vector_info.name)
            if live_vector is None:
                raise SchemaConflictError(f"Missing vector {vector_info.name!r} in live collection")
            if int(live_vector.size) != int(vector_info.size):
                raise SchemaConflictError(
                    f"Vector size mismatch for {vector_info.name!r}: current={live_vector.size}, desired={vector_info.size}"
                )
            if str(live_vector.distance) != str(vector_info.distance):
                raise SchemaConflictError(
                    f"Vector distance mismatch for {vector_info.name!r}: current={live_vector.distance}, desired={vector_info.distance}"
                )

    async def _validate_live_sparse_vectors(self, model: type[QdrantModel]) -> None:
        meta = model.schema_definition()
        collection = await self.client.get_collection(meta.collection_name)
        config = collection.config.params
        live_sparse_vectors = getattr(config.sparse_vectors, "params", config.sparse_vectors)
        live_sparse_vectors_map = live_sparse_vectors if isinstance(live_sparse_vectors, dict) else {}
        for sparse_info in meta.sparse_vector_fields.values():
            if sparse_info.name not in live_sparse_vectors_map:
                raise SchemaConflictError(
                    f"Missing sparse vector {sparse_info.name!r} in live collection"
                )

    def _collect_payload_index_operations(self, model: type[QdrantModel]) -> list[SchemaOperation]:
        meta = model.schema_definition()
        operations: list[SchemaOperation] = []
        for field_name, payload_info in meta.payload_fields.items():
            if payload_info.index is None:
                continue
            index_type = INDEX_TYPES.get(payload_info.index)
            if index_type is None:
                raise SchemaConflictError(f"Unsupported index type: {payload_info.index!r}")
            operations.append(SchemaOperation(operation="create_payload_index", detail=field_name))
        return operations
