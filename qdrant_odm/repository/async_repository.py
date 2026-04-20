from typing import Any, Generic, Sequence, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from qdrant_odm.model.base import QdrantModel
from qdrant_odm.query.compiler import FilterCompiler
from qdrant_odm.query.filters import FilterExpression
from qdrant_odm.query.search import HybridSearchQuery, SearchQuery, SparseVectorInput
from qdrant_odm.repository.base import DEFAULT_RETRIEVE_CHUNK_SIZE, DEFAULT_UPSERT_BATCH_SIZE
from qdrant_odm.repository.result import SearchHit
from qdrant_odm.types import ScrollPage
from qdrant_odm.utils.chunking import chunked

T = TypeVar("T", bound=QdrantModel)


class QdrantRepository(Generic[T]):
    """
    Asynchronous repository for a single Qdrant ODM model.

    This repository provides a typed interface for common Qdrant operations such as:
    - retrieve by id
    - batch retrieval
    - upsert
    - delete
    - payload update
    - existence check
    - count
    - scroll
    - dense or sparse search
    - hybrid search with rank fusion

    The repository is bound to a specific model class and collection schema.
    """

    def __init__(self, client: AsyncQdrantClient, model: type[T]) -> None:
        """
        Initialize the repository for a specific model.

        Args:
            client:
                The asynchronous Qdrant client instance.
            model:
                The ODM model class associated with this repository.
        """
        self.client = client
        self.model = model
        self.meta = model.schema_definition()

    async def get(self, point_id: str | int) -> T | None:
        """
        Retrieve a single point by id and deserialize it into a model instance.

        Args:
            point_id:
                The Qdrant point id.

        Returns:
            The deserialized model instance if found, otherwise None.
        """
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

    async def get_many(
        self,
        ids: Sequence[str | int],
        *,
        chunk_size: int | None = None,
    ) -> list[T]:
        """
        Retrieve multiple points by id and deserialize them into model instances.

        Retrieval is performed in chunks to avoid sending overly large requests.

        Args:
            ids:
                The point ids to retrieve.
            chunk_size:
                Optional batch size override. If omitted, the default retrieve chunk size is used.

        Returns:
            A list of deserialized model instances.
        """
        if not ids:
            return []
        size = chunk_size or DEFAULT_RETRIEVE_CHUNK_SIZE
        results: list[T] = []
        for batch in chunked(list(ids), size):
            records = await self.client.retrieve(
                collection_name=self.meta.collection_name,
                ids=batch,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(
                self.model.from_point(point_id=record.id, payload=record.payload) for record in records
            )
        return results

    async def upsert(self, obj: T, *, vectors: dict[str, Any]) -> None:
        """
        Upsert a single model instance together with its vector data.

        Args:
            obj:
                The model instance to store.
            vectors:
                A mapping of Qdrant named vector keys to vector values.
        """
        point = models.PointStruct(id=obj.model_id(), payload=obj.to_payload(), vector=vectors)
        await self.client.upsert(collection_name=self.meta.collection_name, points=[point])

    async def upsert_many(
        self,
        items: Sequence[tuple[T, dict[str, Any]]],
        *,
        batch_size: int | None = None,
    ) -> None:
        """
        Upsert multiple model instances in batches.

        Args:
            items:
                A sequence of `(model, vectors)` pairs.
            batch_size:
                Optional batch size override. If omitted, the default upsert batch size is used.
        """
        if not items:
            return
        size = batch_size or DEFAULT_UPSERT_BATCH_SIZE
        for batch in chunked(list(items), size):
            points = [
                models.PointStruct(id=obj.model_id(), payload=obj.to_payload(), vector=vectors)
                for obj, vectors in batch
            ]
            await self.client.upsert(collection_name=self.meta.collection_name, points=points)

    async def delete(self, point_id: str | int) -> None:
        """
        Delete a single point by id.

        Args:
            point_id:
                The Qdrant point id to delete.
        """
        selector = models.PointIdsList(points=[point_id])
        await self.client.delete(collection_name=self.meta.collection_name, points_selector=selector)

    async def delete_many(self, ids: Sequence[str | int]) -> None:
        """
        Delete multiple points by id.

        Args:
            ids:
                The point ids to delete.
        """
        if not ids:
            return
        selector = models.PointIdsList(points=list(ids))
        await self.client.delete(collection_name=self.meta.collection_name, points_selector=selector)

    async def set_payload(self, point_id: str | int, payload: dict[str, Any]) -> None:
        """
        Update payload values for a single point.

        Args:
            point_id:
                The Qdrant point id to update.
            payload:
                The payload values to set.
        """
        selector = models.PointIdsList(points=[point_id])
        await self.client.set_payload(
            collection_name=self.meta.collection_name,
            payload=payload,
            points=selector,
        )

    async def exists(self, point_id: str | int) -> bool:
        """
        Check whether a point exists.

        Args:
            point_id:
                The Qdrant point id to check.

        Returns:
            True if the point exists, otherwise False.
        """
        records = await self.client.retrieve(
            collection_name=self.meta.collection_name,
            ids=[point_id],
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)

    async def count(self, filter: FilterExpression | None = None) -> int:
        """
        Count points in the collection, optionally filtered by a query expression.

        Args:
            filter:
                Optional filter expression to restrict the count.

        Returns:
            The number of matching points.
        """
        query_filter = FilterCompiler.compile(filter)
        result = await self.client.count(
            collection_name=self.meta.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        return int(result.count)

    async def scroll(
        self,
        *,
        filter: FilterExpression | None = None,
        limit: int = 100,
        offset: Any | None = None,
        with_vectors: bool = False,
    ) -> ScrollPage[T]:
        """
        Scroll through points in the collection with optional filtering.

        This method returns a page of model instances together with the next offset
        needed to continue scrolling.

        Args:
            filter:
                Optional filter expression to restrict the result set.
            limit:
                Maximum number of points to return in one page.
            offset:
                Optional scroll offset returned from a previous call.
            with_vectors:
                Whether vector data should be included in the underlying Qdrant response.

        Returns:
            A scroll page containing deserialized model instances and the next offset.
        """
        query_filter = FilterCompiler.compile(filter)
        points, next_offset = await self.client.scroll(
            collection_name=self.meta.collection_name,
            scroll_filter=query_filter,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
            limit=limit,
        )
        items = [self.model.from_point(point_id=point.id, payload=point.payload) for point in points]
        return ScrollPage(items=items, next_offset=next_offset)

    async def search(self, query: SearchQuery) -> list[SearchHit[T]]:
        """
        Execute a dense or sparse search query and return typed search hits.

        Args:
            query:
                The search query definition.

        Returns:
            A list of typed search hits containing score, payload, optional vectors,
            and the deserialized model instance.
        """
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
        """
        Execute a hybrid search by combining dense and sparse search results.

        Hybrid retrieval is implemented by:
        1. running a dense search,
        2. running a sparse search,
        3. fusing both ranked lists using reciprocal rank fusion (RRF).

        Args:
            query:
                The hybrid search query definition.

        Returns:
            A fused list of search hits ranked by reciprocal rank fusion.
        """
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
        """
        Convert a search query vector into the Qdrant client query vector format.

        Dense vectors are wrapped as `NamedVector`, while sparse vectors are wrapped
        as `NamedSparseVector`.

        Args:
            query:
                The search query containing the vector input.

        Returns:
            A Qdrant-compatible query vector object.
        """
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
        """
        Fuse dense and sparse ranked results using reciprocal rank fusion (RRF).

        Each hit receives a score contribution of `1 / (k + rank)` from each ranked list.
        Hits appearing in both lists accumulate contributions.

        Args:
            dense_hits:
                Ranked hits from dense retrieval.
            sparse_hits:
                Ranked hits from sparse retrieval.
            limit:
                Maximum number of fused hits to return.
            k:
                RRF smoothing constant.

        Returns:
            A fused and reranked list of search hits.
        """
        rank_scores: dict[Any, float] = {}
        by_id: dict[Any, SearchHit[T]] = {}

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