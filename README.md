# 📦 Qdrant ODM (v0.2)

`qdrant-odm` is a Qdrant-first ODM designed for building production-grade vector search systems with:

- declarative model schema
- explicit payload indexing
- safe schema synchronization (diff → plan → sync)
- Python-native filter DSL
- async repository abstraction
- hybrid retrieval (dense + sparse with RRF)
- batch-optimized operations

---

# 🚀 Installation

```bash
pip install -e ".[dev]"
```

---

# 🚀 Quick Start

```python
from datetime import datetime
from uuid import UUID, uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_odm import (
    PayloadField,
    QdrantModel,
    QdrantODM,
    QdrantRepository,
    SearchQuery,
    SparseVectorField,
    VectorField,
)

class Document(QdrantModel):
    __collection__ = "documents"

    id: UUID
    title: str = PayloadField(index="keyword")
    page: int | None = PayloadField(index="integer")
    category: str = PayloadField(index="keyword")
    created_at: datetime = PayloadField(index="datetime")

    dense = VectorField(name="content_dense", size=3072, distance="Cosine")
    sparse = SparseVectorField(name="content_sparse")

async def run():
    client = AsyncQdrantClient(url="http://localhost:6333")

    odm = QdrantODM(client)
    await odm.sync_schema(Document)

    repo = QdrantRepository[Document](client, Document)

    doc = Document(
        id=uuid4(),
        title="Some Documents",
        page=1,
        category="Some Category",
        created_at=datetime.now(),
    )

    await repo.upsert(
        doc,
        vectors={
            "content_dense": [0.1] * 3072,
            "content_sparse": {"indices": [1, 3], "values": [0.4, 0.7]},
        },
    )
```

---

# 🧠 Core Concepts

## 1. Model = Payload + Schema (Vectors are separate)

- Pydantic fields → payload
- VectorField → schema only
- SparseVectorField → schema only

Vectors are **not stored in the model instance**

```python
await repo.upsert(model, vectors={...})
```

---

## 2. Payload fields vs Indexed fields

```python
title: str                     # stored, NOT indexed
title: PayloadField(index="keyword")  # stored + indexed
```

| Type | Stored | Indexed | Filter | Performance |
|------|--------|---------|--------|------------|
| Normal field | ✅ | ❌ | ✅ | ❌ slow |
| PayloadField | ✅ | ✅ | ✅ | ✅ fast |

---

## 3. Filter DSL

```python
(Document.category == "law") & (Document.page >= 2)
```

Supported:

- `== != > >= < <=`
- `.in_()`
- `.not_in()`
- `.is_null()`
- `.is_not_null()`
- `& | ~`

---

## 4. Repository

```python
repo = QdrantRepository(client, Document)
```

### CRUD

```python
await repo.get(id)
await repo.get_many(ids)
await repo.delete(id)
await repo.exists(id)
```

---

### Upsert

```python
await repo.upsert(model, vectors={...})
```

Batch:

```python
await repo.upsert_many([
    (model1, vectors1),
    (model2, vectors2),
])
```

---

### Scroll

```python
page = await repo.scroll(limit=100)
page.items
page.next_offset
```

---

### Count

```python
await repo.count(filter=...)
```

---

## 5. Search

### Dense

```python
await repo.search(SearchQuery(...))
```

### Sparse

```python
SparseVectorInput(indices=[...], values=[...])
```

### Hybrid (RRF)

```python
await repo.search_hybrid(HybridSearchQuery(...))
```

---

## 6. Schema Sync

### Diff

```python
diff = await odm.schema.diff(Model)
```

### Dry Run

```python
plan = await odm.schema.dry_run(Model)
```

### Sync

```python
await odm.sync_schema(Model)
```

---

# ⚠️ Important Behaviors

## Undeclared payload

- upsert → ONLY model fields
- set_payload → ANY payload allowed

## Index is NOT automatic

Explicit declaration required.

## Schema sync is safe

Will NOT auto-fix:

- vector mismatch
- index type mismatch

## Filter ignores index

Allowed but slow.

## Hybrid search is not native

Uses:

- dense search
- sparse search
- RRF merge

---

# ⚡ Advanced Usage

## Custom batching

```python
await repo.get_many(ids, chunk_size=256)
await repo.upsert_many(items, batch_size=200)
```

## Scroll loop

```python
offset = None
while True:
    page = await repo.scroll(offset=offset)
    if not page.items:
        break
    offset = page.next_offset
```

---

# 🔥 Summary

- Qdrant-first ODM
- strict schema + flexible payload patch
- explicit indexing model
- Python DSL queries
- production-ready async design
