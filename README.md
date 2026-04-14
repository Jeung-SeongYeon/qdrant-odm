## Qdrant ODM MVP

`qdrant-odm` is a Qdrant-first ODM focused on:

- declarative model schema
- collection/index sync
- typed filter DSL
- async repository on top of `AsyncQdrantClient`
- typed search results
- schema dry-run planning
- hybrid retrieval skeleton (dense + sparse RRF)

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

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

    plan = await odm.schema.dry_run(Document)
    print(plan)
```

## Included MVP Modules

- `qdrant_odm/model.py`: `QdrantModel` and metadata collection
- `qdrant_odm/fields.py`: `PayloadField`, `VectorField`, `SparseVectorField`
- `qdrant_odm/query.py`: typed filter expression + compiler + `SearchQuery` / `HybridSearchQuery`
- `qdrant_odm/schema.py`: `SchemaManager.sync` + `plan_sync` + `dry_run`
- `qdrant_odm/repository.py`: async repository methods + `search_hybrid`
- `qdrant_odm/result.py`: typed `SearchHit[T]`
- `qdrant_odm/exceptions.py`: domain exceptions

## Run Tests

```bash
pytest -q
```
