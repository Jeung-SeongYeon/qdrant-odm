from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


class Transaction:
    """
    Asynchronous context manager for buffering Qdrant update operations.

    This class enables a Unit-of-Work pattern by accumulating operations 
    (such as upsert, delete, set_payload) into an internal buffer. 
    When the transaction context exits successfully, it groups these operations 
    by collection and submits them to Qdrant via the `batch_update_points` API.
    
    If an exception occurs within the context, the transaction rolls back 
    by clearing the buffered operations without submitting anything to Qdrant.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        """
        Initialize the Transaction with an asynchronous Qdrant client.

        Args:
            client:
                The asynchronous Qdrant client instance to use for committing operations.
        """
        self.client = client
        self.operations: dict[str, list[models.UpdateOperation]] = {}

    def add_operation(self, collection_name: str, operation: models.UpdateOperation) -> None:
        """
        Add a single Qdrant update operation to the transaction buffer.

        Args:
            collection_name:
                The name of the collection this operation applies to.
            operation:
                The Qdrant UpdateOperation (e.g., UpsertOperation, DeleteOperation) to buffer.
        """
        if collection_name not in self.operations:
            self.operations[collection_name] = []
        self.operations[collection_name].append(operation)

    async def commit(self) -> None:
        """
        Commit all buffered operations to Qdrant.

        Operations are grouped by collection name and submitted via `batch_update_points`.
        After successfully sending all batches, the transaction buffer is cleared.
        """
        for collection_name, ops in self.operations.items():
            if ops:
                await self.client.batch_update_points(
                    collection_name=collection_name,
                    update_operations=ops,
                    wait=True,
                )
        self.operations.clear()

    def rollback(self) -> None:
        """
        Roll back the transaction by clearing all buffered operations.
        """
        self.operations.clear()

    async def __aenter__(self) -> "Transaction":
        self.operations.clear()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        if exc_type is not None:
            self.rollback()
        else:
            await self.commit()
