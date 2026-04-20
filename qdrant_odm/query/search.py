from pydantic import BaseModel, Field
from qdrant_client.http import models

from qdrant_odm.query.filters import FilterExpression


class SparseVectorInput(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)

    def to_qdrant(self) -> models.SparseVector:
        return models.SparseVector(indices=self.indices, values=self.values)


class SearchQuery(BaseModel):
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
    dense_using: str
    dense_vector: list[float]
    sparse_using: str
    sparse_vector: SparseVectorInput
    filter: FilterExpression | None = None
    limit: int = 10
    with_payload: bool = True
    with_vectors: bool = False
    score_threshold: float | None = None
    fusion_k: int = 60

    model_config = {"arbitrary_types_allowed": True}
