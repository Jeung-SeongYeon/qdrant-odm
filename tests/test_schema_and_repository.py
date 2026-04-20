from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from qdrant_client.http import models

from qdrant_odm import (
    HybridSearchQuery,
    PayloadField,
    QdrantModel,
    QdrantRepository,
    SchemaManager,
    SearchHit,
    SparseVectorField,
    SparseVectorInput,
    VectorField,
)


class Document(QdrantModel):
    __collection__ = "documents"
    id: UUID
    title: str = PayloadField(index="keyword")
    created_at: datetime = PayloadField(index="datetime")
    dense = VectorField(name="content_dense", size=4, distance="Cosine")
    sparse = SparseVectorField(name="content_sparse")


@pytest.mark.asyncio
async def test_schema_dry_run_create_collection() -> None:
    client = AsyncMock()
    client.collection_exists.return_value = False
    manager = SchemaManager(client)

    operations = await manager.dry_run(Document)
    operation_names = [operation.operation for operation in operations]

    assert "create_collection" in operation_names
    assert "create_payload_index" in operation_names


@pytest.mark.asyncio
async def test_schema_diff_missing_payload_index() -> None:
    client = AsyncMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={
                    "content_dense": SimpleNamespace(size=4, distance="Cosine"),
                },
                sparse_vectors={"content_sparse": SimpleNamespace()},
            )
        ),
        payload_schema={
            "title": models.PayloadIndexInfo(data_type=models.PayloadSchemaType.KEYWORD, points=0),
        },
    )
    manager = SchemaManager(client)

    diff = await manager.diff(Document)

    assert diff.collection_exists is True
    assert "created_at" in diff.payload_index_missing
    assert not diff.vector_missing
    assert not diff.sparse_missing


@pytest.mark.asyncio
async def test_hybrid_rrf_fusion() -> None:
    client = AsyncMock()
    repository = QdrantRepository(client, Document)

    doc_a = Document(id=uuid4(), title="A", created_at=datetime(2026, 1, 1))
    doc_b = Document(id=uuid4(), title="B", created_at=datetime(2026, 1, 1))
    doc_c = Document(id=uuid4(), title="C", created_at=datetime(2026, 1, 1))

    dense_hits = [
        SearchHit(id=doc_a.id, score=0.9, payload={"title": "A"}, vectors=None, document=doc_a),
        SearchHit(id=doc_b.id, score=0.8, payload={"title": "B"}, vectors=None, document=doc_b),
    ]
    sparse_hits = [
        SearchHit(id=doc_b.id, score=0.7, payload={"title": "B"}, vectors=None, document=doc_b),
        SearchHit(id=doc_c.id, score=0.6, payload={"title": "C"}, vectors=None, document=doc_c),
    ]

    repository.search = AsyncMock(side_effect=[dense_hits, sparse_hits])

    results = await repository.search_hybrid(
        HybridSearchQuery(
            dense_using="content_dense",
            dense_vector=[0.1, 0.2, 0.3, 0.4],
            sparse_using="content_sparse",
            sparse_vector=SparseVectorInput(indices=[1, 3], values=[0.9, 0.7]),
            limit=3,
        )
    )

    assert len(results) == 3
    assert results[0].id == doc_b.id


@pytest.mark.asyncio
async def test_repository_get_returns_model() -> None:
    client = AsyncMock()
    point_id = uuid4()
    client.retrieve.return_value = [SimpleNamespace(id=point_id, payload={"title": "X", "created_at": datetime(2026, 1, 1)})]
    repository = QdrantRepository(client, Document)

    result = await repository.get(point_id)

    assert result is not None
    assert result.id == point_id
    assert result.title == "X"


@pytest.mark.asyncio
async def test_exists_uses_minimal_payload() -> None:
    client = AsyncMock()
    point_id = uuid4()
    client.retrieve.return_value = [SimpleNamespace(id=point_id)]
    repository = QdrantRepository(client, Document)

    found = await repository.exists(point_id)

    assert found is True
    _, kwargs = client.retrieve.call_args
    assert kwargs["with_payload"] is False


@pytest.mark.asyncio
async def test_upsert_many_chunking() -> None:
    client = AsyncMock()
    repository = QdrantRepository(client, Document)
    vectors = {"content_dense": [0.1, 0.2, 0.3, 0.4], "content_sparse": {"indices": [1], "values": [1.0]}}
    items = [
        (
            Document(id=uuid4(), title=f"t{i}", created_at=datetime(2026, 1, 1)),
            vectors,
        )
        for i in range(5)
    ]

    await repository.upsert_many(items, batch_size=2)

    assert client.upsert.await_count == 3
