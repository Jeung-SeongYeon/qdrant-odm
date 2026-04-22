from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from qdrant_odm.exceptions import SchemaConflictError
from qdrant_odm.model.base import QdrantModel
from qdrant_odm.schema.diff import SchemaDiff, compute_schema_diff
from qdrant_odm.schema.planner import SchemaOperation, build_sync_operations
from qdrant_odm.schema.qdrant_schema import build_payload_index_schema, to_distance


class SchemaManager:
    """
    Coordinate schema inspection, planning, and synchronization for ODM models.

    The schema manager is responsible for:
    - computing schema diffs,
    - validating whether differences are safe to reconcile automatically,
    - planning synchronization operations,
    - applying supported schema changes to Qdrant.

    Currently supported synchronization actions include:
    - creating a collection,
    - creating missing payload indexes.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        """
        Initialize the schema manager.

        Args:
            client:
                The asynchronous Qdrant client instance.
        """
        self.client = client

    async def diff(self, model: type[QdrantModel]) -> SchemaDiff:
        """
        Compute the schema diff between the model and the live collection.

        Args:
            model:
                The ODM model class to inspect.

        Returns:
            A structured schema diff.
        """
        return await compute_schema_diff(self.client, model)

    async def plan_sync(self, model: type[QdrantModel]) -> list[SchemaOperation]:
        """
        Build a synchronization plan for a model.

        This method first computes the schema diff, checks for blocking issues,
        and then generates a list of operations needed to reconcile the live schema.

        Args:
            model:
                The ODM model class to synchronize.

        Returns:
            A list of schema operations to execute.

        Raises:
            SchemaConflictError:
                If the diff contains blocking issues that cannot be synchronized automatically.
        """
        schema_diff = await self.diff(model)
        self._raise_if_blocking(schema_diff)
        return build_sync_operations(model, schema_diff)

    async def dry_run(self, model: type[QdrantModel]) -> list[SchemaOperation]:
        """
        Return the synchronization plan without applying it.

        Args:
            model:
                The ODM model class to inspect.

        Returns:
            A list of schema operations that would be executed.
        """
        return await self.plan_sync(model)

    async def sync(self, model: type[QdrantModel]) -> None:
        """
        Apply supported schema synchronization operations for a model.

        This method executes the planned operations sequentially.

        Supported operations currently include:
        - `create_collection`
        - `create_payload_index`

        Args:
            model:
                The ODM model class to synchronize.

        Raises:
            SchemaConflictError:
                If an unsupported payload index type is encountered.
        """
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
                field_schema = build_payload_index_schema(payload_info)
                await self.client.create_payload_index(
                    collection_name=model.collection_name(),
                    field_name=field_name,
                    field_schema=field_schema,
                )

    def _raise_if_blocking(self, schema_diff: SchemaDiff) -> None:
        """
        Raise an error if the schema diff contains blocking issues.

        Blocking issues include:
        - missing dense vectors,
        - dense vector mismatches,
        - missing sparse vectors,
        - payload index type mismatches.

        Args:
            schema_diff:
                The schema diff to validate.

        Raises:
            SchemaConflictError:
                If the diff contains blocking issues.
        """
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
        if schema_diff.payload_index_option_mismatches:
            raise SchemaConflictError("; ".join(schema_diff.payload_index_option_mismatches))

    def _build_vectors_config(self, model: type[QdrantModel]) -> dict[str, models.VectorParams]:
        """
        Build the dense vector configuration for collection creation.

        Args:
            model:
                The ODM model class whose vector definitions should be used.

        Returns:
            A dictionary mapping dense vector names to `VectorParams`.
        """
        meta = model.schema_definition()
        vectors_config: dict[str, models.VectorParams] = {}
        for vector_info in meta.vector_fields.values():
            vectors_config[vector_info.name] = models.VectorParams(
                size=vector_info.size,
                distance=to_distance(vector_info.distance),
                on_disk=vector_info.on_disk,
            )
        return vectors_config

    def _build_sparse_vectors_config(self, model: type[QdrantModel]) -> dict[str, models.SparseVectorParams]:
        """
        Build the sparse vector configuration for collection creation.

        Args:
            model:
                The ODM model class whose sparse vector definitions should be used.

        Returns:
            A dictionary mapping sparse vector names to `SparseVectorParams`.
        """
        meta = model.schema_definition()
        sparse_vectors_config: dict[str, models.SparseVectorParams] = {}
        for sparse_info in meta.sparse_vector_fields.values():
            sparse_vectors_config[sparse_info.name] = models.SparseVectorParams()
        return sparse_vectors_config