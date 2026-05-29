# 📦 Qdrant ODM (v0.3.8)

[한국어(Korean)](./README-KR.md)

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

# 📌 Collection Configuration

You can configure collection-level parameters using `__collection_config__`.

## Modes

### Global (default)
```python
from qdrant_odm import CollectionConfig

class Document(QdrantModel):
    __collection__ = "documents"
    __collection_config__ = CollectionConfig(mode="global")
```

### Multitenant
```python
from qdrant_odm import CollectionConfig, KeywordIndexOptions

class Document(QdrantModel):
    __collection__ = "documents"
    __collection_config__ = CollectionConfig(mode="multitenant")

    tenant_id: str = PayloadField(
        index="keyword",
        keyword=KeywordIndexOptions(is_tenant=True)
    )
```

#### Rules
- Exactly ONE tenant index
- Must be keyword type
- Must set `is_tenant=True`

## Advanced Parameters

`CollectionConfig` supports all Qdrant-native collection configuration parameters, which are applied during collection creation (`sync_schema`):

```python
from qdrant_odm import CollectionConfig
from qdrant_client.http import models

class Document(QdrantModel):
    __collection__ = "documents"
    __collection_config__ = CollectionConfig(
        # Sharding & Replication
        shard_number=2,
        replication_factor=3,
        write_consistency_factor=2,
        
        # Payload & HNSW config
        on_disk_payload=True,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        
        # Optimizers
        optimizers_config=models.OptimizersConfigDiff(deleted_threshold=0.2),
        
        # Quantization
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        ),
    )
```

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

# 📌 Dynamic Payload Field

Use `DynamicPayloadField` to define a schema-less field for storing arbitrary dynamic data. Values stored in this field are flattened into the final Qdrant payload during serialization and are excluded from ODM schema management (such as index creation or diff checks). 

You can annotate the field as either a `dict` or a Pydantic `BaseModel`.

## Dict Example

```python
from qdrant_odm import QdrantModel, PayloadField, DynamicPayloadField

class Document(QdrantModel):
    __collection__ = "documents"

    id: UUID
    title: str = PayloadField()
    
    # Define dynamic payload field annotated as dict
    extra: dict = DynamicPayloadField()
```

```python
doc = Document(
    id=uuid4(),
    title="Schema-less payload",
    extra={"tags": ["ai", "rag"], "views": 42}
)

payload = doc.to_payload()
# {
#     "title": "Schema-less payload",
#     "tags": ["ai", "rag"],
#     "views": 42
# }
```

## BaseModel Example

You can also use a Pydantic `BaseModel` to enforce schema constraints on your dynamic payload while still flattening it:

```python
from pydantic import BaseModel
from qdrant_odm import QdrantModel, PayloadField, DynamicPayloadField

class MetadataExtra(BaseModel):
    tags: list[str]
    views: int

class Document(QdrantModel):
    __collection__ = "documents"

    id: UUID
    title: str = PayloadField()
    extra: MetadataExtra = DynamicPayloadField()
```

```python
doc = Document(
    id=uuid4(),
    title="BaseModel payload",
    extra=MetadataExtra(tags=["ai", "rag"], views=42)
)

payload = doc.to_payload()
# {
#     "title": "BaseModel payload",
#     "tags": ["ai", "rag"],
#     "views": 42
# }

# Automatically reconstructed back into MetadataExtra instance upon from_point
restored = Document.from_point(point_id=doc.id, payload=payload)
print(restored.extra)  # MetadataExtra(tags=['ai', 'rag'], views=42)
```

## Constraints
- A model class can define **only one** `DynamicPayloadField`.
- The field must be annotated as `dict` (or `dict[str, Any]`, etc.) or a Pydantic `BaseModel` subclass.
- Keys in the dynamic payload must not conflict with other static payload fields in the model. If a conflict occurs during serialization, a `ValueError` is raised.
- Values must be a dictionary or a `BaseModel` instance. If not, a `TypeError` is raised.

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

## Transaction (Unit of Work)

Using the `Transaction` context manager, you can safely buffer operations (upsert, delete, set_payload) and submit them in a single batch to Qdrant. If an exception occurs, all buffered operations are safely discarded.

```python
from qdrant_odm import Transaction

async with Transaction(client) as tx:
    await repo.upsert(obj1, vectors={"content_dense": [...]}, tx=tx)
    await repo.set_payload(obj1.id, {"status": "updated"}, tx=tx)
    await repo.delete(obj2.id, tx=tx)
    
    # Operations are automatically committed at the end of the block via `batch_update_points`
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

## Recover from Snapshot

```python
await odm.recover_from_snapshot(
    Model,
    snapshot_path="file:///absolute/path/to/collection.snapshot",
)
```

### Overwrite existing collection
```python
await odm.recover_from_snapshot(
    Model,
    snapshot_path="file:///absolute/path/to/collection.snapshot",
    overwrite=True,
)
```

### Notes
- `snapshot_path` should be a valid snapshot location understood by Qdrant
- Relative paths such as `./test.snapshot` are not supported directly
- Use `overwrite=True` to replace an existing collection before recovery
- After recovery, qdrant-odm runs a schema diff check
- If the recovered collection does not fully match the ODM model schema, a warning is emitted

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
