from dataclasses import dataclass

from qdrant_odm.model.base import QdrantModel
from qdrant_odm.schema.diff import SchemaDiff


@dataclass(slots=True)
class SchemaOperation:
    """
    Declarative schema synchronization operation.

    Each operation represents a single action the schema manager may perform,
    such as creating a collection or creating a payload index.

    Attributes:
        operation:
            The operation name.
        detail:
            Additional operation detail, such as a collection name or field name.
    """

    operation: str
    detail: str


def build_sync_operations(model: type[QdrantModel], diff: SchemaDiff) -> list[SchemaOperation]:
    """
    Build a list of schema synchronization operations from a schema diff.

    The planner currently generates operations for:
    - creating a missing collection,
    - creating missing payload indexes.

    It does not attempt to repair blocking issues such as vector mismatches or
    payload index type mismatches. Those are expected to be handled earlier.

    Args:
        model:
            The ODM model class to synchronize.
        diff:
            The previously computed schema diff.

    Returns:
        A list of schema operations to execute.
    """
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