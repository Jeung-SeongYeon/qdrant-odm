from qdrant_client import AsyncQdrantClient

from qdrant_odm.model import QdrantModel
from qdrant_odm.schema import SchemaManager


class QdrantODM:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client
        self.schema = SchemaManager(client)

    async def sync_schema(self, model: type[QdrantModel]) -> None:
        await self.schema.sync(model)
