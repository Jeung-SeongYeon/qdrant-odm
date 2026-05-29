# 📦 Qdrant ODM (v0.3.8)

[English](./README.md)

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

# 📌 Collection 설정 및 모드 (Collection Configuration)

`__collection_config__`를 사용하여 컬렉션 수준의 설정을 구성할 수 있습니다.

## 모드 (Modes)

### Global (기본값)
```python
from qdrant_odm import CollectionConfig

class Document(QdrantModel):
    __collection__ = "documents"
    __collection_config__ = CollectionConfig(mode="global")
```

### Multitenant (멀티테넌트)
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

#### 규칙
- tenant index는 반드시 1개만 설정 가능
- keyword 타입이어야 함
- `is_tenant=True` 설정이 필수

## 고급 설정 파라미터 (Advanced Parameters)

`CollectionConfig`는 컬렉션 생성(`sync_schema`) 시 적용할 수 있는 Qdrant 네이티브 설정 파라미터들을 모두 지원합니다:

```python
from qdrant_odm import CollectionConfig
from qdrant_client.http import models

class Document(QdrantModel):
    __collection__ = "documents"
    __collection_config__ = CollectionConfig(
        # 샤딩 및 복제 (Sharding & Replication)
        shard_number=2,
        replication_factor=3,
        write_consistency_factor=2,
        
        # 페이로드 및 HNSW 설정
        on_disk_payload=True,
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        
        # 옵티마이저 설정
        optimizers_config=models.OptimizersConfigDiff(deleted_threshold=0.2),
        
        # 양자화 설정 (Quantization)
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        ),
    )
```

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

# 📌 Dynamic Payload Field (동적 페이로드 필드)

`DynamicPayloadField`를 사용하면 임의의 key-value 딕셔너리 데이터를 저장할 수 있는 스키마리스(schema-less) 필드를 정의할 수 있습니다. 이 필드에 저장된 값은 직렬화 시 Qdrant 페이로드의 최상위(root) 레벨로 평탄화(Flatten)되어 저장되며, 인덱스 자동 생성이나 스키마 비교(diff) 등 ODM의 스키마 관리 대상에서 제외됩니다.

필드의 타입은 `dict` 혹은 Pydantic `BaseModel`로 어노테이션할 수 있습니다.

## Dict 사용 예시

```python
from qdrant_odm import QdrantModel, PayloadField, DynamicPayloadField

class Document(QdrantModel):
    __collection__ = "documents"

    id: UUID
    title: str = PayloadField()
    
    # dict 타입으로 정의된 동적 페이로드 필드
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

## BaseModel 사용 예시

Pydantic `BaseModel`을 사용하면 동적 필드에 구조적 스키마 제약을 적용하면서도 Qdrant 페이로드 상에서는 평탄화(Flatten)되도록 구현할 수 있습니다:

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

# from_point 역직렬화 시 자동으로 MetadataExtra 인스턴스로 복원됩니다.
restored = Document.from_point(point_id=doc.id, payload=payload)
print(restored.extra)  # MetadataExtra(tags=['ai', 'rag'], views=42)
```

## 제약 사항
- 하나의 모델 클래스에서는 **최대 1개**의 `DynamicPayloadField`만 정의할 수 있습니다.
- 필드는 반드시 `dict` (또는 `dict[str, Any]` 등) 혹은 Pydantic `BaseModel` 서브클래스 형태로 타입 어노테이션되어야 합니다.
- 동적 필드 내의 key는 모델의 다른 정적 필드명(예: `title`)과 겹치면 안 됩니다. 직렬화(`to_payload()`) 시 중복된 key가 존재하면 `ValueError`가 발생합니다.
- 동적 필드에 설정되는 값은 반드시 딕셔너리(`dict`) 혹은 `BaseModel` 인스턴스 형태여야 합니다. 그렇지 않으면 `TypeError`가 발생합니다.

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

## Transaction (Unit of Work)

`Transaction` 컨텍스트 매니저를 사용하면 여러 작업(upsert, delete, set_payload)을 안전하게 버퍼링하고 단일 배치로 Qdrant에 반영할 수 있습니다. 예외가 발생하면 버퍼링된 작업은 안전하게 폐기(롤백)됩니다.

```python
from qdrant_odm import Transaction

async with Transaction(client) as tx:
    await repo.upsert(obj1, vectors={"content_dense": [...]}, tx=tx)
    await repo.set_payload(obj1.id, {"status": "updated"}, tx=tx)
    await repo.delete(obj2.id, tx=tx)
    
    # 블록이 끝날 때 자동으로 `batch_update_points`를 통해 일괄 반영됩니다.
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
