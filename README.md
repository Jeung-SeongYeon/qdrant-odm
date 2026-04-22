# 📦 Qdrant ODM (v0.3.0)

`qdrant-odm` is a **Qdrant-first ODM** for building production-grade vector search systems.

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

```bash
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
