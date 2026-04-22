from typing import Any, ClassVar, get_args, get_origin, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from qdrant_odm.exceptions import ModelDefinitionError
from qdrant_odm.model.fields import (
    BoolIndexOptions,
    DatetimeIndexOptions,
    FloatIndexOptions,
    GeoIndexOptions,
    IntegerIndexOptions,
    KeywordIndexOptions,
    PayloadFieldInfo,
    SparseVectorField,
    SparseVectorFieldInfo,
    TextIndexOptions,
    UuidIndexOptions,
    VectorField,
    VectorFieldInfo,
)
from qdrant_odm.model.metadata import CollectionConfig, ModelMetadata
from qdrant_odm.query.expressions import FieldExpr

_ALLOWED_DISTANCE_VALUES = {"Cosine", "Euclid", "Dot", "Manhattan"}
_ALLOWED_COLLECTION_MODES = {"global", "multitenant"}


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

    Requirements for every concrete model:
    - `__collection__` must be defined
    - the configured id field must exist
    - the id field must be `UUID` or `int`
    - at least one dense `VectorField` must be declared
    - sparse vector fields are optional

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
    __collection_config__: ClassVar[CollectionConfig] = CollectionConfig()
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
                - if no dense vector field is declared,
                - if vector names are duplicated.
        """
        super().__pydantic_init_subclass__(**kwargs)
        if cls.__name__ == "QdrantModel":
            return

        collection_name = getattr(cls, "__collection__", "")
        if not collection_name:
            raise ModelDefinitionError(f"{cls.__name__} must define __collection__")

        collection_config = getattr(cls, "__collection_config__", CollectionConfig())
        if not isinstance(collection_config, CollectionConfig):
            raise ModelDefinitionError(
                f"{cls.__name__}.__collection_config__ must be a CollectionConfig instance"
            )

        if collection_config.mode not in _ALLOWED_COLLECTION_MODES:
            raise ModelDefinitionError(
                f"{cls.__name__} has unsupported collection mode "
                f"{collection_config.mode!r}. Allowed: {sorted(_ALLOWED_COLLECTION_MODES)!r}"
            )

        id_field = getattr(cls, "__id_field__", "id")
        if id_field not in cls.model_fields:
            raise ModelDefinitionError(f"{cls.__name__} must define id field {id_field!r}")

        field_info = cls.model_fields[id_field]
        annotation = field_info.annotation

        def _is_valid_id_type(tp: Any) -> bool:
            if tp in (int, UUID):
                return True

            origin = get_origin(tp)
            if origin is Union:
                return all(_is_valid_id_type(arg) for arg in get_args(tp))

            return False

        if not _is_valid_id_type(annotation):
            raise ModelDefinitionError(
                f"{cls.__name__}.{id_field} must be of type UUID or int (got {annotation!r})"
            )

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

        _validate_payload_index_options(cls, payload_fields)
        tenant_field = _validate_collection_mode(cls, collection_config, payload_fields)

        vector_fields: dict[str, VectorFieldInfo] = {}
        sparse_vector_fields: dict[str, SparseVectorFieldInfo] = {}
        named_vectors: set[str] = set()

        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, VectorField):
                if attr_value.info.name in named_vectors:
                    raise ModelDefinitionError(
                        f"{cls.__name__} has duplicated vector name {attr_value.info.name!r}"
                    )
                if attr_value.info.distance not in _ALLOWED_DISTANCE_VALUES:
                    raise ModelDefinitionError(
                        f"{cls.__name__}.{attr_name} has unsupported distance "
                        f"{attr_value.info.distance!r}. Allowed: {sorted(_ALLOWED_DISTANCE_VALUES)!r}"
                    )
                named_vectors.add(attr_value.info.name)
                vector_fields[attr_name] = attr_value.info
            elif isinstance(attr_value, SparseVectorField):
                if attr_value.info.name in named_vectors:
                    raise ModelDefinitionError(
                        f"{cls.__name__} has duplicated vector name {attr_value.info.name!r}"
                    )
                named_vectors.add(attr_value.info.name)
                sparse_vector_fields[attr_name] = attr_value.info

        if not vector_fields:
            raise ModelDefinitionError(
                f"{cls.__name__} must define at least one dense VectorField"
            )

        cls.__odm_meta__ = ModelMetadata(
            collection_name=collection_name,
            id_field=id_field,
            collection_config=collection_config,
            tenant_field=tenant_field,
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

        UUID values are converted to strings because Qdrant point ids are passed
        to the client as string or integer identifiers.
        """
        value = getattr(self, self.__odm_meta__.id_field)
        if isinstance(value, UUID):
            return str(value)
        return value

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


def _validate_payload_index_options(
    model_cls: type[QdrantModel],
    payload_fields: dict[str, PayloadFieldInfo],
) -> None:
    """
    Validate that only the option object matching the declared `index`
    is provided for each payload field.
    """
    option_map: dict[str, tuple[str, type[Any]]] = {
        "keyword": ("keyword", KeywordIndexOptions),
        "integer": ("integer", IntegerIndexOptions),
        "float": ("float_", FloatIndexOptions),
        "bool": ("bool_", BoolIndexOptions),
        "geo": ("geo", GeoIndexOptions),
        "datetime": ("datetime", DatetimeIndexOptions),
        "text": ("text", TextIndexOptions),
        "uuid": ("uuid", UuidIndexOptions),
    }

    option_attr_names = tuple(attr for attr, _ in option_map.values())

    for field_name, payload_info in payload_fields.items():
        if payload_info.index is None:
            for attr_name in option_attr_names:
                if getattr(payload_info, attr_name) is not None:
                    raise ModelDefinitionError(
                        f"{model_cls.__name__}.{field_name} defines payload index options "
                        f"but has no `index=` declared"
                    )
            continue

        if payload_info.index not in option_map:
            raise ModelDefinitionError(
                f"{model_cls.__name__}.{field_name} uses unsupported payload index "
                f"type {payload_info.index!r}"
            )

        expected_attr_name, _ = option_map[payload_info.index]

        for attr_name in option_attr_names:
            value = getattr(payload_info, attr_name)
            if value is None:
                continue
            if attr_name != expected_attr_name:
                raise ModelDefinitionError(
                    f"{model_cls.__name__}.{field_name} uses index={payload_info.index!r} "
                    f"but provided incompatible option set {attr_name!r}"
                )


def _validate_collection_mode(
    model_cls: type[QdrantModel],
    collection_config: CollectionConfig,
    payload_fields: dict[str, PayloadFieldInfo],
) -> str | None:
    """
    Validate collection-level constraints.

    For `multitenant` mode, exactly one keyword payload index with
    `is_tenant=True` must exist.

    Returns:
        The tenant field name for multitenant models, otherwise None.
    """
    if collection_config.mode == "global":
        return None

    tenant_fields: list[str] = []
    for field_name, payload_info in payload_fields.items():
        if payload_info.index != "keyword":
            continue
        if payload_info.keyword is None:
            continue
        if payload_info.keyword.is_tenant is True:
            tenant_fields.append(field_name)

    if not tenant_fields:
        raise ModelDefinitionError(
            f"{model_cls.__name__} uses multitenant collection mode but does not define "
            f"a keyword payload index with keyword.is_tenant=True"
        )

    if len(tenant_fields) > 1:
        raise ModelDefinitionError(
            f"{model_cls.__name__} uses multitenant collection mode but defines multiple "
            f"tenant keyword indexes: {tenant_fields!r}"
        )

    return tenant_fields[0]