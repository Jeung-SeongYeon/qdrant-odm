from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from qdrant_odm.exceptions import ModelDefinitionError
from qdrant_odm.fields import PayloadFieldInfo, SparseVectorField, SparseVectorFieldInfo, VectorField, VectorFieldInfo
from qdrant_odm.query import FieldExpr


@dataclass(slots=True)
class ModelMetadata:
    collection_name: str
    id_field: str
    payload_fields: dict[str, PayloadFieldInfo] = field(default_factory=dict)
    vector_fields: dict[str, VectorFieldInfo] = field(default_factory=dict)
    sparse_vector_fields: dict[str, SparseVectorFieldInfo] = field(default_factory=dict)


class QdrantModelMeta(type(BaseModel)):
    def __getattr__(cls, name: str) -> FieldExpr:
        if name.startswith("__"):
            raise AttributeError(f"{cls.__name__!s} has no attribute {name!r}")
        odm_meta = cls.__dict__.get("__odm_meta__")
        if odm_meta and name in odm_meta.payload_fields:
            return FieldExpr(field_name=name)
        raise AttributeError(f"{cls.__name__!s} has no attribute {name!r}")


class QdrantModel(BaseModel, metaclass=QdrantModelMeta):
    __collection__: ClassVar[str]
    __id_field__: ClassVar[str] = "id"
    __odm_meta__: ClassVar[ModelMetadata]
    model_config = ConfigDict(ignored_types=(VectorField, SparseVectorField))

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if cls.__name__ == "QdrantModel":
            return

        collection_name = getattr(cls, "__collection__", "")
        if not collection_name:
            raise ModelDefinitionError(f"{cls.__name__} must define __collection__")

        id_field = getattr(cls, "__id_field__", "id")
        if id_field not in cls.model_fields:
            raise ModelDefinitionError(f"{cls.__name__} must define id field {id_field!r}")

        payload_fields: dict[str, PayloadFieldInfo] = {}
        for field_name, field_info in cls.model_fields.items():
            if field_name == id_field:
                continue
            json_schema_extra = field_info.json_schema_extra or {}
            payload_meta = json_schema_extra.get("qdrant_payload")
            if isinstance(payload_meta, PayloadFieldInfo):
                payload_fields[field_name] = payload_meta
                continue
            payload_fields[field_name] = PayloadFieldInfo()

        vector_fields: dict[str, VectorFieldInfo] = {}
        sparse_vector_fields: dict[str, SparseVectorFieldInfo] = {}
        named_vectors: set[str] = set()

        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, VectorField):
                if attr_value.info.name in named_vectors:
                    raise ModelDefinitionError(
                        f"{cls.__name__} has duplicated vector name {attr_value.info.name!r}"
                    )
                named_vectors.add(attr_value.info.name)
                vector_fields[attr_name] = attr_value.info
            if isinstance(attr_value, SparseVectorField):
                if attr_value.info.name in named_vectors:
                    raise ModelDefinitionError(
                        f"{cls.__name__} has duplicated vector name {attr_value.info.name!r}"
                    )
                named_vectors.add(attr_value.info.name)
                sparse_vector_fields[attr_name] = attr_value.info

        cls.__odm_meta__ = ModelMetadata(
            collection_name=collection_name,
            id_field=id_field,
            payload_fields=payload_fields,
            vector_fields=vector_fields,
            sparse_vector_fields=sparse_vector_fields,
        )

    @classmethod
    def collection_name(cls) -> str:
        return cls.__odm_meta__.collection_name

    @classmethod
    def schema_definition(cls) -> ModelMetadata:
        return cls.__odm_meta__

    def model_id(self) -> Any:
        return getattr(self, self.__odm_meta__.id_field)

    def to_payload(self) -> dict[str, Any]:
        exclude = {self.__odm_meta__.id_field}
        return self.model_dump(mode="python", by_alias=True, exclude=exclude)

    @classmethod
    def from_point(cls, *, point_id: Any, payload: dict[str, Any] | None) -> "QdrantModel":
        source_payload = payload or {}
        data = dict(source_payload)
        data[cls.__odm_meta__.id_field] = point_id
        return cls.model_validate(data)
