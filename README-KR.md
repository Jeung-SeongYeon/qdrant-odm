# 📦 Qdrant ODM (v0.3.4)

`qdrant-odm`은 프로덕션 환경의 벡터 검색 시스템 구축을 위한 **Qdrant 전용 ODM(Object Document Mapper)** 입니다.

---

# 왜 qdrant-odm인가?

Qdrant를 직접 사용하는 것은 강력하지만, 프로젝트 규모가 커질수록 코드가 점점 복잡해집니다.

대표적인 문제들:

- payload 필드를 문자열로 반복 작성
- collection 및 index를 수동으로 관리
- Qdrant low-level filter 작성의 번거로움
- schema / query / repository 로직이 뒤섞임

`qdrant-odm`은 Qdrant의 특성을 유지하면서도 더 구조화된 방식으로 사용할 수 있도록 도와줍니다.

---

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

---

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

---

## 무엇이 달라졌나?

- 모델 기반으로 schema를 한 번만 정의
- index 설정을 필드 옆에 명시
- Python 표현식 기반 filter DSL
- repository 패턴으로 CRUD 및 검색 로직 통일
- schema sync로 collection 설정 자동화

---

# 🚀 주요 기능

- Pydantic 기반 선언형 schema
- 세밀한 payload index 제어
- Collection 모드 (global / multitenant)
- 안전한 schema sync (diff → plan → sync)
- Python-native filter DSL
- Async repository abstraction
- Hybrid 검색 (dense + sparse + RRF)
- batch 최적화 처리

---

# 🚀 설치

```bash
pip install qdrant-odmx
```

개발 환경:

```bash
git clone https://github.com/Jeung-SeongYeon/qdrant-odm
cd qdrant-odm
pip install -e ".[dev]"
```

---

# 🧠 아키텍처

```
Model → Metadata → SchemaManager → Qdrant
     ↘ Query DSL → Compiler → Filter
     ↘ Repository → CRUD / Search
```

---

# 📌 모델 정의

```python
class Document(QdrantModel):
    __collection__ = "documents"

    id: UUID
    title: str = PayloadField(index="keyword")
    created_at: datetime = PayloadField(index="datetime")

    dense = VectorField(name="content_dense", size=3072)
```

---

# 📌 Collection 모드

## Global
```python
__collection_config__ = CollectionConfig(mode="global")
```

## Multitenant
```python
tenant_id: str = PayloadField(
    index="keyword",
    keyword=KeywordIndexOptions(is_tenant=True)
)
```

규칙:
- tenant index는 반드시 1개
- keyword 타입
- is_tenant=True 필수

---

# 📌 Vector 정의

```python
VectorField(
    name="content_dense",
    size=3072,
    distance="Cosine"
)
```

지원 거리:

- Cosine
- Euclid
- Dot
- Manhattan

---

# 📌 Payload Index 옵션

```python
page: int = PayloadField(
    index="integer",
    integer=IntegerIndexOptions(
        lookup=True,
        range=True,
        on_disk=True
    )
)
```

지원 타입:

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

지원 연산:

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

# 🔍 검색

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

(RRF 기반)

---

# 🧬 Schema Sync

```python
await odm.schema.diff(Model)
await odm.schema.dry_run(Model)
await odm.sync_schema(Model)
```

---

# 📦 Snapshot 복구

```python
await odm.recover_from_snapshot(
    Model,
    snapshot_path="file:///absolute/path/to/collection.snapshot",
)
```

overwrite:

```python
await odm.recover_from_snapshot(
    Model,
    snapshot_path="file:///absolute/path/to/collection.snapshot",
    overwrite=True,
)
```

---

# ⚠️ 동작 규칙

변경 불가:

- vector schema
- index 타입
- index 옵션

가능:

- collection 생성
- index 생성

---

# ⚡ 성능 팁

- filter 필드는 반드시 index
- batch 연산 사용
- 대용량 데이터는 scroll 활용
- hybrid search로 recall 향상

---

# 🔥 요약

- Qdrant 전용 ODM
- 강한 schema 보장
- 멀티테넌트 지원
- 프로덕션 수준 async 설계
