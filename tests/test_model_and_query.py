from datetime import datetime
from uuid import UUID, uuid4

from qdrant_odm import PayloadField, QdrantModel, VectorField
from qdrant_odm.query import ComparisonExpr, LogicalExpr


class Document(QdrantModel):
    __collection__ = "documents"
    id: UUID
    title: str = PayloadField(index="keyword")
    page: int | None = PayloadField(index="integer")
    created_at: datetime = PayloadField(index="datetime")
    dense = VectorField(name="content_dense", size=4, distance="Cosine")


def test_model_metadata_collection() -> None:
    meta = Document.schema_definition()
    assert meta.collection_name == "documents"
    assert meta.id_field == "id"
    assert "title" in meta.payload_fields
    assert "dense" in meta.vector_fields


def test_model_payload_conversion() -> None:
    document = Document(
        id=uuid4(),
        title="hello",
        page=2,
        created_at=datetime(2026, 1, 1),
    )
    payload = document.to_payload()
    assert "id" not in payload
    assert payload["title"] == "hello"
    restored = Document.from_point(point_id=document.id, payload=payload)
    assert restored.id == document.id


def test_filter_expression_building() -> None:
    expr = (Document.title == "law") & (Document.page >= 2)
    assert isinstance(expr, LogicalExpr)
    assert len(expr.values) == 2
    assert isinstance(expr.values[0], ComparisonExpr)
