from dataclasses import dataclass, field

from qdrant_odm.model.fields import PayloadFieldInfo, SparseVectorFieldInfo, VectorFieldInfo


@dataclass(slots=True)
class ModelMetadata:
    collection_name: str
    id_field: str
    payload_fields: dict[str, PayloadFieldInfo] = field(default_factory=dict)
    vector_fields: dict[str, VectorFieldInfo] = field(default_factory=dict)
    sparse_vector_fields: dict[str, SparseVectorFieldInfo] = field(default_factory=dict)
