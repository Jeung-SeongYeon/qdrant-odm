from typing import Any, Union, Literal
from pydantic import BaseModel, Field
from qdrant_client.http import models

from qdrant_odm.query.filters import FilterExpression


class SparseVectorInput(BaseModel):
    """
    Input model for a sparse query vector.

    This model represents sparse vector data in index-value form and provides
    a helper method to convert it into the Qdrant client sparse vector model.
    """

    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    def to_qdrant(self) -> models.SparseVector:
        """
        Convert this sparse vector input into a Qdrant sparse vector object.

        Returns:
            A `models.SparseVector` instance compatible with the Qdrant client.
        """
        return models.SparseVector(indices=self.indices, values=self.values)


class SearchQuery(BaseModel):
    """
    Query model for a single-vector search request.

    This model represents the parameters needed to perform a search against
    a specific named vector using an optional filter expression.

    Attributes:
        using:
            The Qdrant named vector key to search against.
        vector:
            The query vector, either dense or sparse.
        filter:
            Optional filter expression to restrict matched points.
        limit:
            Maximum number of results to return.
        offset:
            Optional pagination offset.
        with_payload:
            Whether payload data should be included in the response.
        with_vectors:
            Whether vector data should be included in the response.
        score_threshold:
            Optional minimum score threshold for matched results.
    """

    using: str
    vector: list[float] | SparseVectorInput
    filter: FilterExpression | None = None
    limit: int = 10
    offset: int | None = None
    with_payload: bool = True
    with_vectors: bool = False
    score_threshold: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class HybridSearchQuery(BaseModel):
    """
    Query model for a hybrid search request combining dense and sparse vectors.

    This model is intended for search workflows that fuse dense and sparse retrieval
    signals into a single ranked result set.

    Attributes:
        dense_using:
            The Qdrant named dense vector key.
        dense_vector:
            The dense query vector.
        sparse_using:
            The Qdrant named sparse vector key.
        sparse_vector:
            The sparse query vector input.
        filter:
            Optional filter expression to restrict matched points.
        limit:
            Maximum number of results to return.
        with_payload:
            Whether payload data should be included in the response.
        with_vectors:
            Whether vector data should be included in the response.
        score_threshold:
            Optional minimum score threshold for matched results.
        fusion_k:
            Fusion parameter used by the hybrid ranking strategy.
    """

    dense_using: str
    dense_vector: list[float]
    sparse_using: str
    sparse_vector: SparseVectorInput
    filter: FilterExpression | None = None
    limit: int = 10
    with_payload: bool = True
    with_vectors: bool = False
    score_threshold: float | None = None
    fusion_k: int | None = None
    fusion: Literal["RRF", "DBSF"] = "RRF"

    model_config = {"arbitrary_types_allowed": True}


class Prefetch(BaseModel):
    """
    ODM wrapper for Qdrant's Prefetch query.
    """
    query: Any | None = None
    using: str | None = None
    filter: FilterExpression | None = None
    limit: int | None = None
    score_threshold: float | None = None
    prefetch: Union["Prefetch", list["Prefetch"], None] = None
    params: Any | None = None

    model_config = {"arbitrary_types_allowed": True}

    def to_qdrant(self, model: Any | None = None) -> models.Prefetch:
        """
        Convert this Prefetch wrapper into a native Qdrant models.Prefetch instance.
        """
        from qdrant_odm.query.compiler import FilterCompiler

        # Compile filter if present
        compiled_filter = (
            FilterCompiler.compile(self.filter, model=model)
            if self.filter is not None
            else None
        )

        # Resolve query type
        raw_query = self.query
        if isinstance(self.query, SparseVectorInput):
            raw_query = self.query.to_qdrant()

        # Resolve nested prefetch queries recursively
        nested_prefetch: Any = None
        if self.prefetch is not None:
            if isinstance(self.prefetch, list):
                nested_prefetch = [p.to_qdrant(model=model) for p in self.prefetch]
            else:
                nested_prefetch = self.prefetch.to_qdrant(model=model)

        return models.Prefetch(
            query=raw_query,
            using=self.using,
            filter=compiled_filter,
            limit=self.limit,
            score_threshold=self.score_threshold,
            prefetch=nested_prefetch,
            params=self.params,
        )


Prefetch.model_rebuild()