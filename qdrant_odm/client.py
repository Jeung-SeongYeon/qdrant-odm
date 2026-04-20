from qdrant_client import AsyncQdrantClient

from qdrant_odm.model.base import QdrantModel
from qdrant_odm.schema.sync import SchemaManager


class QdrantODM:
    """
    High-level entry point for the qdrant-odm package.

    This class provides a lightweight façade over shared package-level operations,
    such as schema synchronization. It wraps an `AsyncQdrantClient` instance and
    exposes convenience access to the schema manager.

    Attributes:
        client:
            The underlying asynchronous Qdrant client.
        schema:
            The schema manager bound to the same client.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        """
        Initialize the ODM façade with an asynchronous Qdrant client.

        Args:
            client:
                The asynchronous Qdrant client instance to use across the package.
        """
        self.client = client
        self.schema = SchemaManager(client)

    async def sync_schema(self, model: type[QdrantModel]) -> None:
        """
        Synchronize the live Qdrant schema for a model.

        This is a convenience wrapper around the schema manager.

        Args:
            model:
                The ODM model class whose schema should be synchronized.
        """
        await self.schema.sync(model)