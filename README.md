## Qdrant ODM (v0.2)

`qdrant-odm` is a Qdrant-first ODM focused on:

- declarative model schema
- collection and payload index sync with **schema diff** and dry-run planning
- typed filter DSL
- async repository on top of `AsyncQdrantClient`
- typed search results and hybrid retrieval skeleton (dense + sparse RRF)
- **batched** `get_many` / `upsert_many` for large workloads

## Package layout

```text
qdrant_odm/
├── __init__.py
├── client.py
├── exceptions.py
├── types.py
├── model/
│   ├── base.py
│   ├── fields.py
│   ├── metadata.py
│   ├── serializer.py
│   └── registry.py
├── schema/
│   ├── planner.py
│   ├── sync.py
│   ├── diff.py
│   └── qdrant_schema.py
├── query/
│   ├── expressions.py
│   ├── filters.py
│   ├── search.py
│   ├── operators.py
│   └── compiler.py
├── repository/
│   ├── base.py
│   ├── async_repository.py
│   └── result.py
└── utils/
    ├── typing.py
    ├── inspect.py
    └── chunking.py
```

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

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
    is_deleted: bool = PayloadField(default=False, index="bool")

    dense = VectorField(name="content_dense", size=3072, distance="Cosine")
    sparse = SparseVectorField(name="content_sparse")


async def run() -> None:
    client = AsyncQdrantClient(url="http://localhost:6333")
    odm = QdrantODM(client)
    await odm.sync_schema(Document)

    repo = QdrantRepository[Document](client, Document)
    document = Document(
        id=uuid4(),
        title="제10조 관리비",
        page=3,
        category="law",
        created_at=datetime.now(),
        is_deleted=False,
    )

    await repo.upsert(
        document,
        vectors={
            "content_dense": [0.1] * 3072,
            "content_sparse": {"indices": [1, 5], "values": [0.3, 0.8]},
        },
    )

    hits = await repo.search(
        SearchQuery(
            using="content_dense",
            vector=[0.2] * 3072,
            filter=(Document.category == "law") & (Document.page >= 2),
            limit=10,
        )
    )

    for hit in hits:
        print(hit.score, hit.document.title)

    diff = await odm.schema.diff(Document)
    print(diff.payload_index_missing)

    plan = await odm.schema.dry_run(Document)
    print(plan)

    page = await repo.scroll(filter=Document.category == "law", limit=50)
    print(len(page.items), page.next_offset)

    total = await repo.count(filter=Document.is_deleted == False)
    print("count", total)

    await repo.set_payload(document.id, {"page": 4})
    await repo.delete(document.id)
```

## v0.2 highlights

| Area | Behavior |
|------|-----------|
| Schema | `SchemaManager.diff()` returns `SchemaDiff` (vectors, sparse vectors, payload indexes). `sync()` / `dry_run()` / `plan_sync()` use the same planner; blocking issues raise `SchemaConflictError`. |
| Repository | `scroll()` returns `ScrollPage[T]` with `next_offset` for pagination. `get_many(..., chunk_size=...)`, `upsert_many(..., batch_size=...)` use `utils.chunking`. `exists()` uses `retrieve` with `with_payload=False`. |
| Utilities | `chunked()` for splitting ID lists and upsert batches. |

## Run tests

```bash
pytest -q
```
