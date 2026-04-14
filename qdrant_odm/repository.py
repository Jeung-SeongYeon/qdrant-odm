from typing import Any, Generic, Sequence, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from qdrant_odm.model import QdrantModel
from qdrant_odm.query import FilterCompiler, HybridSearchQuery, SearchQuery, SparseVectorInput
from qdrant_odm.result import SearchHit

T = TypeVar("T", bound=QdrantModel)


class QdrantRepository(Generic[T]):
    def __init__(self, client: AsyncQdrantClient, model: type[T]) -> None:
        self.client = client
        self.model = model
        self.meta = model.schema_definition()

    async def get(self, point_id: str | int) -> T | None:
        records = await self.client.retrieve(
            collection_name=self.meta.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        record = records[0]
        return self.model.from_point(point_id=record.id, payload=record.payload)

    async def get_many(self, ids: Sequence[str | int]) -> list[T]:
        if not ids:
            return []
        records = await self.client.retrieve(
            collection_name=self.meta.collection_name,
            ids=list(ids),
            with_payload=True,
            with_vectors=False,
        )
        return [self.model.from_point(point_id=record.id, payload=record.payload) for record in records]

    async def upsert(self, obj: T, *, vectors: dict[str, Any]) -> None:
        point = models.PointStruct(id=obj.model_id(), payload=obj.to_payload(), vector=vectors)
        await self.client.upsert(collection_name=self.meta.collection_name, points=[point])

    async def upsert_many(self, items: Sequence[tuple[T, dict[str, Any]]]) -> None:
        if not items:
            return
        points = [
            models.PointStruct(id=obj.model_id(), payload=obj.to_payload(), vector=vectors)
            for obj, vectors in items
        ]
        await self.client.upsert(collection_name=self.meta.collection_name, points=points)

    async def delete(self, point_id: str | int) -> None:
        selector = models.PointIdsList(points=[point_id])
        await self.client.delete(collection_name=self.meta.collection_name, points_selector=selector)

    async def delete_many(self, ids: Sequence[str | int]) -> None:
        if not ids:
            return
        selector = models.PointIdsList(points=list(ids))
        await self.client.delete(collection_name=self.meta.collection_name, points_selector=selector)

    async def set_payload(self, point_id: str | int, payload: dict[str, Any]) -> None:
        selector = models.PointIdsList(points=[point_id])
        await self.client.set_payload(
            collection_name=self.meta.collection_name,
            payload=payload,
            points=selector,
        )

    async def exists(self, point_id: str | int) -> bool:
        return await self.get(point_id) is not None

    async def count(self, filter: Any | None = None) -> int:
        query_filter = FilterCompiler.compile(filter)
        result = await self.client.count(
            collection_name=self.meta.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(result.count)

    async def scroll(self, *, filter: Any | None = None, limit: int = 100) -> list[T]:
        query_filter = FilterCompiler.compile(filter)
        points, _ = await self.client.scroll(
            collection_name=self.meta.collection_name,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        return [self.model.from_point(point_id=point.id, payload=point.payload) for point in points]

    async def search(self, query: SearchQuery) -> list[SearchHit[T]]:
        search_result = await self.client.search(
            collection_name=self.meta.collection_name,
            query_vector=self._compile_query_vector(query),
            query_filter=FilterCompiler.compile(query.filter),
            limit=query.limit,
            offset=query.offset,
            with_payload=query.with_payload,
            with_vectors=query.with_vectors,
            score_threshold=query.score_threshold,
        )
        hits: list[SearchHit[T]] = []
        for point in search_result:
            payload = dict(point.payload or {})
            document = self.model.from_point(point_id=point.id, payload=payload)
            hits.append(
                SearchHit[T](
                    id=point.id,
                    score=float(point.score),
                    payload=payload,
                    vectors=point.vector,
                    document=document,
                )
            )
        return hits

    async def search_hybrid(self, query: HybridSearchQuery) -> list[SearchHit[T]]:
        dense_hits = await self.search(
            SearchQuery(
                using=query.dense_using,
                vector=query.dense_vector,
                filter=query.filter,
                limit=query.limit,
                with_payload=query.with_payload,
                with_vectors=query.with_vectors,
                score_threshold=query.score_threshold,
            )
        )
        sparse_hits = await self.search(
            SearchQuery(
                using=query.sparse_using,
                vector=query.sparse_vector,
                filter=query.filter,
                limit=query.limit,
                with_payload=query.with_payload,
                with_vectors=query.with_vectors,
                score_threshold=query.score_threshold,
            )
        )
        return self._fuse_hits_rrf(dense_hits=dense_hits, sparse_hits=sparse_hits, limit=query.limit, k=query.fusion_k)

    def _compile_query_vector(
        self, query: SearchQuery
    ) -> models.NamedVector | models.NamedSparseVector | list[float]:
        if isinstance(query.vector, SparseVectorInput):
            return models.NamedSparseVector(name=query.using, vector=query.vector.to_qdrant())
        return models.NamedVector(name=query.using, vector=query.vector)

    def _fuse_hits_rrf(
        self,
        *,
        dense_hits: list[SearchHit[T]],
        sparse_hits: list[SearchHit[T]],
        limit: int,
        k: int,
    ) -> list[SearchHit[T]]:
        rank_scores: dict[str | int, float] = {}
        by_id: dict[str | int, SearchHit[T]] = {}

        for index, hit in enumerate(dense_hits, start=1):
            rank_scores[hit.id] = rank_scores.get(hit.id, 0.0) + (1.0 / (k + index))
            by_id[hit.id] = hit

        for index, hit in enumerate(sparse_hits, start=1):
            rank_scores[hit.id] = rank_scores.get(hit.id, 0.0) + (1.0 / (k + index))
            by_id.setdefault(hit.id, hit)

        ranked_ids = sorted(rank_scores.keys(), key=lambda point_id: rank_scores[point_id], reverse=True)
        fused_hits: list[SearchHit[T]] = []
        for point_id in ranked_ids[:limit]:
            hit = by_id[point_id]
            fused_hits.append(
                SearchHit[T](
                    id=hit.id,
                    score=rank_scores[point_id],
                    payload=hit.payload,
                    vectors=hit.vectors,
                    document=hit.document,
                )
            )
        return fused_hits
