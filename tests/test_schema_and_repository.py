from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from qdrant_client.http import models

from qdrant_odm import (
    CollectionConfig,
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
    point_id = uuid4()
    mock_point = models.ScoredPoint(
        id=point_id,
        version=1,
        score=0.99,
        payload={"title": "Hybrid native result", "created_at": "2026-01-01T00:00:00Z"},
        vector=None,
    )
    client.query_points.return_value = models.QueryResponse(points=[mock_point])

    repository = QdrantRepository(client, Document)

    results = await repository.search_hybrid(
        HybridSearchQuery(
            dense_using="content_dense",
            dense_vector=[0.1, 0.2, 0.3, 0.4],
            sparse_using="content_sparse",
            sparse_vector=SparseVectorInput(indices=[1, 3], values=[0.9, 0.7]),
            limit=3,
        )
    )

    assert len(results) == 1
    assert results[0].id == point_id
    assert results[0].score == 0.99

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert kwargs["collection_name"] == "documents"
    assert isinstance(kwargs["query"], models.FusionQuery)
    assert kwargs["query"].fusion == models.Fusion.RRF

    # Assert prefetches
    prefetch_list = kwargs["prefetch"]
    assert len(prefetch_list) == 2
    assert isinstance(prefetch_list[0], models.Prefetch)
    assert prefetch_list[0].using == "content_dense"
    assert prefetch_list[0].query == [0.1, 0.2, 0.3, 0.4]

    assert isinstance(prefetch_list[1], models.Prefetch)
    assert prefetch_list[1].using == "content_sparse"
    assert isinstance(prefetch_list[1].query, models.SparseVector)
    assert prefetch_list[1].query.indices == [1, 3]
    assert prefetch_list[1].query.values == [0.9, 0.7]


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


@pytest.mark.asyncio
async def test_schema_sync_with_custom_collection_config() -> None:
    quantization = models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))
    hnsw = models.HnswConfigDiff(m=16, ef_construct=100)
    optimizers = models.OptimizersConfigDiff(deleted_threshold=0.2)

    class CustomConfigDoc(QdrantModel):
        __collection__ = "custom_config_docs"
        __collection_config__ = CollectionConfig(
            quantization_config=quantization,
            on_disk_payload=True,
            hnsw_config=hnsw,
            optimizers_config=optimizers,
            shard_number=2,
            replication_factor=3,
            write_consistency_factor=2,
        )
        id: UUID
        title: str = PayloadField(index="keyword")
        dense = VectorField(name="content_dense", size=4, distance="Cosine")

    client = AsyncMock()
    client.collection_exists.return_value = False
    manager = SchemaManager(client)

    await manager.sync(CustomConfigDoc)

    client.create_collection.assert_called_once_with(
        collection_name="custom_config_docs",
        vectors_config={"content_dense": models.VectorParams(size=4, distance=models.Distance.COSINE, on_disk=None)},
        sparse_vectors_config={},
        quantization_config=quantization,
        on_disk_payload=True,
        hnsw_config=hnsw,
        optimizers_config=optimizers,
        shard_number=2,
        replication_factor=3,
        write_consistency_factor=2,
    )


@pytest.mark.asyncio
async def test_repository_search_calls_query_points() -> None:
    client = AsyncMock()
    point_id = uuid4()
    mock_point = models.ScoredPoint(
        id=point_id,
        version=1,
        score=0.95,
        payload={"title": "Test Point", "created_at": "2026-01-01T00:00:00Z"},
        vector=None,
    )
    client.query_points.return_value = models.QueryResponse(points=[mock_point])

    repository = QdrantRepository(client, Document)

    from qdrant_odm.query.search import SearchQuery
    query = SearchQuery(
        using="content_dense",
        vector=[0.1, 0.2, 0.3, 0.4],
        limit=5,
    )

    results = await repository.search(query)

    assert len(results) == 1
    assert results[0].id == point_id
    assert results[0].score == 0.95
    assert results[0].payload["title"] == "Test Point"
    assert results[0].document.title == "Test Point"

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert kwargs["collection_name"] == "documents"
    assert kwargs["query"] == [0.1, 0.2, 0.3, 0.4]
    assert kwargs["using"] == "content_dense"
    assert kwargs["limit"] == 5


@pytest.mark.asyncio
async def test_repository_search_sparse_calls_query_points() -> None:
    client = AsyncMock()
    point_id = uuid4()
    mock_point = models.ScoredPoint(
        id=point_id,
        version=1,
        score=0.85,
        payload={"title": "Sparse Point", "created_at": "2026-01-01T00:00:00Z"},
        vector=None,
    )
    client.query_points.return_value = models.QueryResponse(points=[mock_point])

    repository = QdrantRepository(client, Document)

    from qdrant_odm.query.search import SearchQuery, SparseVectorInput
    query = SearchQuery(
        using="content_sparse",
        vector=SparseVectorInput(indices=[1, 2], values=[0.5, 0.6]),
        limit=2,
    )

    results = await repository.search(query)

    assert len(results) == 1
    assert results[0].id == point_id
    assert results[0].score == 0.85

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert kwargs["collection_name"] == "documents"
    assert isinstance(kwargs["query"], models.SparseVector)
    assert kwargs["query"].indices == [1, 2]
    assert kwargs["query"].values == [0.5, 0.6]
    assert kwargs["using"] == "content_sparse"


@pytest.mark.asyncio
async def test_repository_query_calls_query_points() -> None:
    client = AsyncMock()
    point_id = uuid4()
    mock_point = models.ScoredPoint(
        id=point_id,
        version=1,
        score=0.99,
        payload={"title": "Hybrid Result", "created_at": "2026-01-01T00:00:00Z"},
        vector=None,
    )
    client.query_points.return_value = models.QueryResponse(points=[mock_point])

    repository = QdrantRepository(client, Document)

    prefetch_list = [
        models.Prefetch(
            query=[0.1, 0.2, 0.3, 0.4],
            using="content_dense",
            limit=10,
        )
    ]

    results = await repository.query(
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=prefetch_list,
        limit=5,
    )

    assert len(results) == 1
    assert results[0].id == point_id
    assert results[0].score == 0.99

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert kwargs["collection_name"] == "documents"
    assert isinstance(kwargs["query"], models.FusionQuery)
    assert kwargs["prefetch"] == prefetch_list
    assert kwargs["limit"] == 5


def test_prefetch_wrapper_to_native() -> None:
    from qdrant_odm import Prefetch, SparseVectorInput

    # 1. Basic Conversion
    p = Prefetch(
        query=[0.1, 0.2],
        using="dense",
        limit=5,
    )
    native = p.to_qdrant()
    assert isinstance(native, models.Prefetch)
    assert native.query == [0.1, 0.2]
    assert native.using == "dense"
    assert native.limit == 5

    # 2. Sparse Vector & Filter Conversion
    p_sparse = Prefetch(
        query=SparseVectorInput(indices=[0, 1], values=[1.0, 2.0]),
        using="sparse",
        filter=(Document.title == "filter_test"),
    )
    native_sparse = p_sparse.to_qdrant(model=Document)
    assert isinstance(native_sparse.query, models.SparseVector)
    assert native_sparse.query.indices == [0, 1]
    assert native_sparse.query.values == [1.0, 2.0]
    assert isinstance(native_sparse.filter, models.Filter)

    # 3. Nested Prefetch Conversion
    p_nested = Prefetch(
        prefetch=Prefetch(query=[0.5, 0.6], using="dense_nested")
    )
    native_nested = p_nested.to_qdrant()
    assert isinstance(native_nested.prefetch, models.Prefetch)
    assert native_nested.prefetch.query == [0.5, 0.6]
    assert native_nested.prefetch.using == "dense_nested"


@pytest.mark.asyncio
async def test_hybrid_search_with_filter() -> None:
    client = AsyncMock()
    client.query_points.return_value = models.QueryResponse(points=[])
    repository = QdrantRepository(client, Document)

    # Search with a filter
    await repository.search_hybrid(
        HybridSearchQuery(
            dense_using="content_dense",
            dense_vector=[0.1, 0.2, 0.3, 0.4],
            sparse_using="content_sparse",
            sparse_vector=SparseVectorInput(indices=[1], values=[1.0]),
            filter=(Document.title == "matching_title"),
            limit=5,
        )
    )

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    # The filter should compile to query_filter and be passed to query_points
    assert isinstance(kwargs["query_filter"], models.Filter)
    assert kwargs["query_filter"].must[0].key == "title"
    assert kwargs["query_filter"].must[0].match.value == "matching_title"


@pytest.mark.asyncio
async def test_prefetch_wrapper_in_hybrid_query() -> None:
    client = AsyncMock()
    client.query_points.return_value = models.QueryResponse(points=[])
    repository = QdrantRepository(client, Document)

    from qdrant_odm import Prefetch

    # Pass the ODM Prefetch wrappers to query()
    await repository.query(
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=[
            Prefetch(query=[0.1, 0.2], using="content_dense"),
            Prefetch(query=SparseVectorInput(indices=[0], values=[0.5]), using="content_sparse"),
        ],
        limit=5,
    )

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], models.FusionQuery)

    # Verify both prefetches are converted to Qdrant native Prefetch objects
    prefetches = kwargs["prefetch"]
    assert len(prefetches) == 2
    assert isinstance(prefetches[0], models.Prefetch)
    assert prefetches[0].query == [0.1, 0.2]
    assert prefetches[0].using == "content_dense"

    assert isinstance(prefetches[1], models.Prefetch)
    assert isinstance(prefetches[1].query, models.SparseVector)
    assert prefetches[1].query.indices == [0]
    assert prefetches[1].using == "content_sparse"


@pytest.mark.asyncio
async def test_hybrid_search_legacy_fusion_k_60() -> None:
    client = AsyncMock()
    doc_a = Document(id=uuid4(), title="A", created_at=datetime(2026, 1, 1))
    doc_b = Document(id=uuid4(), title="B", created_at=datetime(2026, 1, 1))

    dense_mock_response = models.QueryResponse(points=[
        models.ScoredPoint(id=doc_a.id, version=1, score=0.9, payload={"title": "A", "created_at": "2026-01-01T00:00:00Z"}),
        models.ScoredPoint(id=doc_b.id, version=1, score=0.8, payload={"title": "B", "created_at": "2026-01-01T00:00:00Z"}),
    ])
    sparse_mock_response = models.QueryResponse(points=[
        models.ScoredPoint(id=doc_b.id, version=1, score=0.7, payload={"title": "B", "created_at": "2026-01-01T00:00:00Z"}),
    ])

    client.query_points.side_effect = [dense_mock_response, sparse_mock_response]

    repository = QdrantRepository(client, Document)

    from unittest.mock import patch
    with patch.object(repository, "_fuse_hits_rrf", wraps=repository._fuse_hits_rrf) as mock_fuse:
        results = await repository.search_hybrid(
            HybridSearchQuery(
                dense_using="content_dense",
                dense_vector=[0.1, 0.2, 0.3, 0.4],
                sparse_using="content_sparse",
                sparse_vector=SparseVectorInput(indices=[1], values=[1.0]),
                limit=3,
                fusion_k=60,
            )
        )

        mock_fuse.assert_called_once()
        _, kwargs = mock_fuse.call_args
        assert kwargs["k"] == 60

    assert len(results) == 2
    assert results[0].id == doc_b.id
    assert results[1].id == doc_a.id

    assert client.query_points.call_count == 2
    for call in client.query_points.call_args_list:
        _, kw = call
        assert not isinstance(kw.get("query"), models.FusionQuery)


@pytest.mark.asyncio
async def test_hybrid_search_legacy_fusion_k_10() -> None:
    client = AsyncMock()
    doc_a = Document(id=uuid4(), title="A", created_at=datetime(2026, 1, 1))
    doc_b = Document(id=uuid4(), title="B", created_at=datetime(2026, 1, 1))

    dense_mock_response = models.QueryResponse(points=[
        models.ScoredPoint(id=doc_a.id, version=1, score=0.9, payload={"title": "A", "created_at": "2026-01-01T00:00:00Z"}),
        models.ScoredPoint(id=doc_b.id, version=1, score=0.8, payload={"title": "B", "created_at": "2026-01-01T00:00:00Z"}),
    ])
    sparse_mock_response = models.QueryResponse(points=[
        models.ScoredPoint(id=doc_b.id, version=1, score=0.7, payload={"title": "B", "created_at": "2026-01-01T00:00:00Z"}),
    ])

    client.query_points.side_effect = [dense_mock_response, sparse_mock_response]

    repository = QdrantRepository(client, Document)

    from unittest.mock import patch
    with patch.object(repository, "_fuse_hits_rrf", wraps=repository._fuse_hits_rrf) as mock_fuse:
        results = await repository.search_hybrid(
            HybridSearchQuery(
                dense_using="content_dense",
                dense_vector=[0.1, 0.2, 0.3, 0.4],
                sparse_using="content_sparse",
                sparse_vector=SparseVectorInput(indices=[1], values=[1.0]),
                limit=3,
                fusion_k=10,
            )
        )

        mock_fuse.assert_called_once()
        _, kwargs = mock_fuse.call_args
        assert kwargs["k"] == 10

    assert len(results) == 2
    assert results[0].id == doc_b.id
    assert abs(results[0].score - 0.17424) < 1e-4
    assert abs(results[1].score - 0.09090) < 1e-4


@pytest.mark.asyncio
async def test_hybrid_search_native_route_no_python_rrf() -> None:
    client = AsyncMock()
    mock_point = models.ScoredPoint(
        id=uuid4(),
        version=1,
        score=0.99,
        payload={"title": "Native result", "created_at": "2026-01-01T00:00:00Z"},
        vector=None,
    )
    client.query_points.return_value = models.QueryResponse(points=[mock_point])

    repository = QdrantRepository(client, Document)

    from unittest.mock import patch
    with patch.object(repository, "_fuse_hits_rrf") as mock_fuse:
        await repository.search_hybrid(
            HybridSearchQuery(
                dense_using="content_dense",
                dense_vector=[0.1, 0.2, 0.3, 0.4],
                sparse_using="content_sparse",
                sparse_vector=SparseVectorInput(indices=[1], values=[1.0]),
                limit=3,
                fusion_k=None,
            )
        )

        mock_fuse.assert_not_called()

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], models.FusionQuery)


@pytest.mark.asyncio
async def test_hybrid_search_native_dbsf() -> None:
    client = AsyncMock()
    mock_point = models.ScoredPoint(
        id=uuid4(),
        version=1,
        score=0.99,
        payload={"title": "Native DBSF result", "created_at": "2026-01-01T00:00:00Z"},
        vector=None,
    )
    client.query_points.return_value = models.QueryResponse(points=[mock_point])

    repository = QdrantRepository(client, Document)

    from unittest.mock import patch
    with patch.object(repository, "_fuse_hits_rrf") as mock_fuse:
        await repository.search_hybrid(
            HybridSearchQuery(
                dense_using="content_dense",
                dense_vector=[0.1, 0.2, 0.3, 0.4],
                sparse_using="content_sparse",
                sparse_vector=SparseVectorInput(indices=[1], values=[1.0]),
                limit=3,
                fusion="DBSF",
            )
        )

        mock_fuse.assert_not_called()

    client.query_points.assert_called_once()
    _, kwargs = client.query_points.call_args
    assert isinstance(kwargs["query"], models.FusionQuery)
    assert kwargs["query"].fusion == models.Fusion.DBSF
