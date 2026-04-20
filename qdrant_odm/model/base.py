from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from qdrant_odm.exceptions import ModelDefinitionError
from qdrant_odm.model.fields import (
    PayloadFieldInfo,
    SparseVectorField,
    SparseVectorFieldInfo,
    VectorField,
    VectorFieldInfo,
)
from qdrant_odm.model.metadata import ModelMetadata
from qdrant_odm.query.expressions import FieldExpr


class QdrantModelMeta(type(BaseModel)):
    """
    Metaclass for Qdrant ODM models.

    This metaclass enables class-level field access for query DSL expressions.
    For example, `Document.title == "foo"` can be translated into a `FieldExpr`
    even though `title` is not defined as a normal class attribute.

    Only payload fields collected in `__odm_meta__` are exposed this way.
    """

    def __getattr__(cls, name: str) -> FieldExpr:
        """
        Return a query field expression for a payload field name.

        This is used to support class-based query syntax such as:
            Document.category == "law"

        Raises:
            AttributeError: If the attribute is not a known payload field.
        """
        if name.startswith("__"):
            raise AttributeError(f"{cls.__name__!s} has no attribute {name!r}")
        odm_meta = cls.__dict__.get("__odm_meta__")
        if odm_meta and name in odm_meta.payload_fields:
            return FieldExpr(field_name=name)
        raise AttributeError(f"{cls.__name__!s} has no attribute {name!r}")


class QdrantModel(BaseModel, metaclass=QdrantModelMeta):
    """
    Base class for all Qdrant ODM models.

    A subclass of `QdrantModel` represents the payload schema of a Qdrant collection
    together with metadata describing:
    - the collection name,
    - the id field,
    - payload field options,
    - dense vector definitions,
    - sparse vector definitions.

    During subclass initialization, ODM metadata is collected automatically and stored
    in `__odm_meta__`.
    """

    __collection__: ClassVar[str]
    __id_field__: ClassVar[str] = "id"
    __odm_meta__: ClassVar[ModelMetadata]
    model_config = ConfigDict(ignored_types=(VectorField, SparseVectorField))

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """
        Build ODM metadata when a concrete model subclass is declared.

        This hook validates required model settings and collects:
        - collection name,
        - id field,
        - payload field metadata,
        - dense vector field metadata,
        - sparse vector field metadata.

        Raises:
            ModelDefinitionError:
                - if `__collection__` is missing,
                - if the configured id field does not exist,
                - if vector names are duplicated.
        """
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
        """
        Return the Qdrant collection name associated with this model.
        """
        return cls.__odm_meta__.collection_name

    @classmethod
    def schema_definition(cls) -> ModelMetadata:
        """
        Return the collected ODM metadata for this model.
        """
        return cls.__odm_meta__

    def model_id(self) -> Any:
        """
        Return the value of the configured model id field.

        This value is intended to be used as the Qdrant point id.
        """
        return getattr(self, self.__odm_meta__.id_field)

    def to_payload(self) -> dict[str, Any]:
        """
        Serialize the model into a Qdrant payload dictionary.

        The configured id field is excluded because the point id is stored separately
        from the payload in Qdrant.
        """
        exclude = {self.__odm_meta__.id_field}
        return self.model_dump(mode="python", by_alias=True, exclude=exclude)

    @classmethod
    def from_point(cls, *, point_id: Any, payload: dict[str, Any] | None) -> "QdrantModel":
        """
        Reconstruct a model instance from a Qdrant point id and payload.

        Args:
            point_id: The Qdrant point id.
            payload: The Qdrant payload dictionary. If None, an empty payload is used.

        Returns:
            A validated model instance with the point id injected into the configured id field.
        """
        source_payload = payload or {}
        data = dict(source_payload)
        data[cls.__odm_meta__.id_field] = point_id
        return cls.model_validate(data)