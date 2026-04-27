from datetime import datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from qdrant_client.http import models

from qdrant_odm import (
    QdrantModel,
    QdrantRepository,
    Transaction,
    PayloadField,
    VectorField,
)


class DummyModel(QdrantModel):
    __collection__ = "dummy"
    id: UUID
    title: str = PayloadField(index="keyword")
    created_at: datetime = PayloadField(index="datetime")
    dense = VectorField(name="content_dense", size=4, distance="Cosine")


@pytest.mark.asyncio
async def test_transaction_commit() -> None:
    client = AsyncMock()
    repo = QdrantRepository(client, DummyModel)
    
    obj = DummyModel(id=uuid4(), title="Test", created_at=datetime(2026, 1, 1))
    vectors = {"content_dense": [0.1, 0.2, 0.3, 0.4]}

    async with Transaction(client) as tx:
        await repo.upsert(obj, vectors=vectors, tx=tx)
        await repo.delete(obj.id, tx=tx)
        await repo.set_payload(obj.id, {"title": "Updated"}, tx=tx)

    assert client.upsert.await_count == 0
    assert client.delete.await_count == 0
    assert client.set_payload.await_count == 0

    assert client.batch_update_points.await_count == 1
    _, kwargs = client.batch_update_points.call_args
    assert kwargs["collection_name"] == "dummy"
    operations = kwargs["update_operations"]
    assert len(operations) == 3
    assert isinstance(operations[0], models.UpsertOperation)
    assert isinstance(operations[1], models.DeleteOperation)
    assert isinstance(operations[2], models.SetPayloadOperation)


@pytest.mark.asyncio
async def test_transaction_rollback_on_exception() -> None:
    client = AsyncMock()
    repo = QdrantRepository(client, DummyModel)
    
    obj = DummyModel(id=uuid4(), title="Test", created_at=datetime(2026, 1, 1))
    vectors = {"content_dense": [0.1, 0.2, 0.3, 0.4]}

    class TestError(Exception):
        pass

    try:
        async with Transaction(client) as tx:
            await repo.upsert(obj, vectors=vectors, tx=tx)
            raise TestError("Simulated error")
    except TestError:
        pass

    assert client.batch_update_points.await_count == 0
    assert len(tx.operations) == 0


@pytest.mark.asyncio
async def test_transaction_multiple_collections() -> None:
    client = AsyncMock()
    
    class ModelA(QdrantModel):
        __collection__ = "col_a"
        id: UUID
        dense = VectorField(name="content_dense", size=4, distance="Cosine")
        
    class ModelB(QdrantModel):
        __collection__ = "col_b"
        id: UUID
        dense = VectorField(name="content_dense", size=4, distance="Cosine")
        
    repo_a = QdrantRepository(client, ModelA)
    repo_b = QdrantRepository(client, ModelB)

    async with Transaction(client) as tx:
        await repo_a.delete(uuid4(), tx=tx)
        await repo_b.delete(uuid4(), tx=tx)
        await repo_a.delete(uuid4(), tx=tx)

    assert client.batch_update_points.await_count == 2
    
    calls = client.batch_update_points.call_args_list
    collections_called = [call.kwargs["collection_name"] for call in calls]
    assert "col_a" in collections_called
    assert "col_b" in collections_called
    
    for call in calls:
        if call.kwargs["collection_name"] == "col_a":
            assert len(call.kwargs["update_operations"]) == 2
        elif call.kwargs["collection_name"] == "col_b":
            assert len(call.kwargs["update_operations"]) == 1
