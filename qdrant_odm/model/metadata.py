from typing import Any, Literal
from dataclasses import dataclass, field

from qdrant_client.http import models

from qdrant_odm.model.fields import PayloadFieldInfo, SparseVectorFieldInfo, VectorFieldInfo

CollectionModeLiteral = Literal["global", "multitenant"]


@dataclass(slots=True)
class CollectionConfig:
    """
    Collection-level ODM configuration.
    """

    mode: CollectionModeLiteral = "global"
    quantization_config: models.QuantizationConfig | None = None
    on_disk_payload: bool | None = None
    hnsw_config: models.HnswConfigDiff | None = None
    optimizers_config: models.OptimizersConfigDiff | None = None
    shard_number: int | None = None
    replication_factor: int | None = None
    write_consistency_factor: int | None = None


@dataclass(slots=True)
class ModelMetadata:
    """
    Collected ODM metadata for a concrete model class.

    This object represents the compiled schema information derived from a
    `QdrantModel` subclass definition.

    Attributes:
        collection_name:
            The target Qdrant collection name.
        id_field:
            The model field used as the Qdrant point id.
        payload_fields:
            Mapping of model payload field names to payload metadata.
        vector_fields:
            Mapping of model attribute names to dense vector metadata.
        sparse_vector_fields:
            Mapping of model attribute names to sparse vector metadata.
    """

    collection_name: str
    id_field: str
    collection_config: CollectionConfig = field(default_factory=CollectionConfig)
    tenant_field: str | None = None
    payload_fields: dict[str, PayloadFieldInfo] = field(default_factory=dict)
    dynamic_payload_fields: dict[str, Any] = field(default_factory=dict)
    vector_fields: dict[str, VectorFieldInfo] = field(default_factory=dict)
    sparse_vector_fields: dict[str, SparseVectorFieldInfo] = field(default_factory=dict)