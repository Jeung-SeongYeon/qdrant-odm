# 📦 Qdrant ODM (v0.3.3)

`qdrant-odm` is a **Qdrant-first ODM** for building production-grade vector search systems.

---

# Why qdrant-odm?

Working with raw Qdrant is powerful, but it gets verbose as your project grows.

Typical pain points:

- Repeating payload field names as raw strings
- Manually managing collection and index setup
- Writing filters in low-level Qdrant syntax
- Mixing schema, query, and repository logic in application code

`qdrant-odm` keeps Qdrant native, but gives you a more structured way to work with it.

## Before: raw Qdrant client

```python
from uuid import uuid4
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

client = AsyncQdrantClient(url="http://localhost:6333")

await client.create_collection(
    collection_name="documents",
    vectors_config={
        "content_dense": qm.VectorParams(size=3072, distance=qm.Distance.COSINE)
    }
)

await client.create_payload_index(
    collection_name="documents",
    field_name="title",
    field_schema=qm.PayloadSchemaType.KEYWORD
)

await client.upsert(
    collection_name="documents",
    points=[
        qm.PointStruct(
            id=str(uuid4()),
            vector={"content_dense": [0.1] * 3072},
            payload={
                "title": "Qdrant ODM",
                "category": "tech",
                "page": 1,
            },
        )
    ],
)

results = await client.query_points(
    collection_name="documents",
    query=[0.1] * 3072,
    using="content_dense",
    query_filter=qm.Filter(
        must=[
            qm.FieldCondition(
                key="category",
                match=qm.MatchValue(value="tech")
            ),
            qm.FieldCondition(
                key="page",
                range=qm.Range(gte=1)
            ),
        ]
    ),
    limit=10,
)
```

## After: qdrant-odm

```python
from uuid import UUID, uuid4
from qdrant_odm import (
    QdrantModel,
    PayloadField,
    VectorField,
    QdrantODM,
    QdrantRepository,
    SearchQuery,
)

class Document(QdrantModel):
    __collection__ = "documents"

    id: UUID
    title: str = PayloadField(index="keyword")
    category: str = PayloadField(index="keyword")
    page: int = PayloadField(index="integer")

    dense = VectorField(name="content_dense", size=3072)

odm = QdrantODM(client)
await odm.sync_schema(Document)

repo = QdrantRepository(client, Document)

await repo.upsert(
    Document(
        id=uuid4(),
        title="Qdrant ODM",
        category="tech",
        page=1,
    ),
    vectors={
        "content_dense": [0.1] * 3072,
    },
)

results = await repo.search(
    SearchQuery(
        vector=[0.1] * 3072,
        using="content_dense",
        filter=(Document.category == "tech") & (Document.page >= 1),
        limit=10,
    )
)
```

## What changes?
- Schema is defined once in the model
- Index configuration lives next to fields
- Filters use Python expressions instead of raw payload strings
- Repositories keep CRUD and search logic consistent
- Schema sync reduces collection setup boilerplate

---

# 🚀 Features

- Declarative schema (Pydantic-based)
- Explicit payload indexing with fine-grained control
- Collection modes (global / multitenant)
- Safe schema sync (diff → plan → sync)
- Python-native filter DSL
- Async repository abstraction
- Hybrid retrieval (dense + sparse with RRF)
- Batch optimized operations

---

# 🚀 Installation

## Installation

```bash
pip install qdrant-odmx
```

## Development
```python
git clone https://github.com/Jeung-SeongYeon/qdrant-odm
cd qdrant-odm
pip install -e ".[dev]"
```

---

# 🧠 Architecture Overview

```
Model → Metadata → SchemaManager → Qdrant
     ↘ Query DSL → Compiler → Filter
     ↘ Repository → CRUD / Search
```

---

# 📌 Model Definition

## Basic Model

```python
from uuid import UUID
from datetime import datetime
from qdrant_odm import QdrantModel, PayloadField, VectorField

class Document(QdrantModel):
    __collection__ = "documents"

    id: UUID
    title: str = PayloadField(index="keyword")
    created_at: datetime = PayloadField(index="datetime")

    dense = VectorField(name="content_dense", size=3072)
```

---

# 📌 Collection Modes

## Global (default)

```python
__collection_config__ = CollectionConfig(mode="global")
```

## Multitenant

```python
from qdrant_odm import CollectionConfig, KeywordIndexOptions

__collection_config__ = CollectionConfig(mode="multitenant")

tenant_id: str = PayloadField(
    index="keyword",
    keyword=KeywordIndexOptions(is_tenant=True)
)
```

### Rules

- Exactly ONE tenant index
- Must be keyword
- Must set `is_tenant=True`

---

# 📌 Vector Definition

```python
VectorField(
    name="content_dense",
    size=3072,
    distance="Cosine"
)
```

### Supported distances

- Cosine
- Euclid
- Dot
- Manhattan

---

# 📌 Payload Index Options

```python
from qdrant_odm import IntegerIndexOptions

page: int = PayloadField(
    index="integer",
    integer=IntegerIndexOptions(
        lookup=True,
        range=True,
        on_disk=True
    )
)
```

Supported:

- keyword
- integer
- float
- bool
- geo
- datetime
- text
- uuid

---

# 🔍 Query DSL

```python
(Document.category == "law") & (Document.page >= 2)
```

Supports:

- == != > >= < <=
- in_ / not_in
- is_null / is_not_null
- &, |, ~

---

# 📦 Repository

```python
repo = QdrantRepository(client, Document)
```

## CRUD

```python
await repo.get(id)
await repo.delete(id)
await repo.exists(id)
```

## Batch

```python
await repo.upsert_many([...])
```

## Scroll

```python
page = await repo.scroll()
```

---

# 🔍 Search

## Dense

```python
await repo.search(SearchQuery(...))
```

## Sparse

```python
SparseVectorInput(indices=[...], values=[...])
```

## Hybrid

```python
await repo.search_hybrid(HybridSearchQuery(...))
```

Uses RRF internally.

---

# 🧬 Schema Sync

## Diff

```python
await odm.schema.diff(Model)
```

## Dry Run

```python
await odm.schema.dry_run(Model)
```

## Sync

```python
await odm.sync_schema(Model)
```

---

# ⚠️ Behavior Rules

## Safe Sync

Will NOT modify:

- vector schema
- index type
- index options

Only:

- create collection
- create missing indexes

---

## Payload Behavior

- upsert → model fields only
- set_payload → free form

---

## Filtering

- Works without index
- Slower without index

---

# ⚡ Performance Tips

- Always index filter fields
- Use batch operations
- Use scroll for large datasets
- Use hybrid search for recall boost

---

# 🔥 Summary

- Qdrant-native ODM
- strict schema guarantees
- multitenant ready
- production-ready async design
