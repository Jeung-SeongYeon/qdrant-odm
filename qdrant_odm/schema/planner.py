from dataclasses import dataclass

from qdrant_odm.model.base import QdrantModel
from qdrant_odm.schema.diff import SchemaDiff


@dataclass(slots=True)
class SchemaOperation:
    operation: str
    detail: str


def build_sync_operations(model: type[QdrantModel], diff: SchemaDiff) -> list[SchemaOperation]:
    operations: list[SchemaOperation] = []
    meta = model.schema_definition()

    if not diff.collection_exists:
        operations.append(SchemaOperation(operation="create_collection", detail=meta.collection_name))
        for field_name, payload_info in meta.payload_fields.items():
            if payload_info.index is None:
                continue
            operations.append(SchemaOperation(operation="create_payload_index", detail=field_name))
        return operations

    for field_name in diff.payload_index_missing:
        operations.append(SchemaOperation(operation="create_payload_index", detail=field_name))

    return operations
