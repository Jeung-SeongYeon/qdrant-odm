from datetime import datetime
from uuid import UUID, uuid4

from qdrant_odm import DynamicPayloadField, PayloadField, QdrantModel, VectorField
from qdrant_odm.query import ComparisonExpr, LogicalExpr
import pytest


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


def test_dynamic_payload_field() -> None:
    class DynamicDoc(QdrantModel):
        __collection__ = "dynamic_docs"
        id: UUID
        title: str = PayloadField()
        extra: dict = DynamicPayloadField()
        dense = VectorField(name="content_dense", size=4, distance="Cosine")

    # 1. Metadata verification
    meta = DynamicDoc.schema_definition()
    assert "extra" in meta.dynamic_payload_fields
    assert "extra" not in meta.payload_fields
    assert "title" in meta.payload_fields

    # 2. Serialization (flattening) verification
    doc = DynamicDoc(
        id=uuid4(),
        title="hello",
        extra={"custom_field": 123, "another": "val"},
    )
    payload = doc.to_payload()
    assert "extra" not in payload
    assert payload["title"] == "hello"
    assert payload["custom_field"] == 123
    assert payload["another"] == "val"

    # 3. Type verification (must be dict)
    doc_invalid = DynamicDoc(id=uuid4(), title="hello", extra={"ok": 1})
    doc_invalid.extra = "not a dict"
    with pytest.raises(TypeError, match="Dynamic payload field 'extra' must be a dict"):
        doc_invalid.to_payload()

    # 4. Conflict verification
    doc_conflict = DynamicDoc(
        id=uuid4(),
        title="hello",
        extra={"title": "conflict"},
    )
    with pytest.raises(ValueError, match="Dynamic payload key 'title' conflicts with existing payload field"):
        doc_conflict.to_payload()

    # 5. Deserialization (reconstruction) verification
    point_id = uuid4()
    restored = DynamicDoc.from_point(point_id=point_id, payload=payload)
    assert restored.id == point_id
    assert restored.title == "hello"
    assert restored.extra == {"custom_field": 123, "another": "val"}
