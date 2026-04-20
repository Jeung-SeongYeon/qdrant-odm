from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field
from qdrant_client.http import models

PayloadSchemaLiteral = Literal[
    "keyword",
    "integer",
    "float",
    "bool",
    "geo",
    "datetime",
    "text",
    "uuid",
]

TokenizerLiteral = Literal["word", "whitespace", "prefix", "multilingual"]


@dataclass(slots=True)
class KeywordIndexOptions:
    """
    Extra options for a keyword payload index.
    """

    is_tenant: bool | None = None
    on_disk: bool | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class IntegerIndexOptions:
    """
    Extra options for an integer payload index.
    """

    lookup: bool | None = None
    range: bool | None = None
    is_principal: bool | None = None
    on_disk: bool | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class FloatIndexOptions:
    """
    Extra options for a float payload index.
    """

    is_principal: bool | None = None
    on_disk: bool | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class BoolIndexOptions:
    """
    Extra options for a bool payload index.
    """

    on_disk: bool | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class GeoIndexOptions:
    """
    Extra options for a geo payload index.
    """

    on_disk: bool | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class DatetimeIndexOptions:
    """
    Extra options for a datetime payload index.
    """

    is_principal: bool | None = None
    on_disk: bool | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class TextIndexOptions:
    """
    Extra options for a text payload index.
    """

    tokenizer: TokenizerLiteral | None = None
    min_token_len: int | None = None
    max_token_len: int | None = None
    lowercase: bool | None = None
    ascii_folding: bool | None = None
    phrase_matching: bool | None = None
    stopwords: Any | None = None
    on_disk: bool | None = None
    stemmer: models.SnowballParams | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class UuidIndexOptions:
    """
    Extra options for a UUID payload index.
    """

    is_tenant: bool | None = None
    on_disk: bool | None = None
    enable_hnsw: bool | None = None


@dataclass(slots=True)
class PayloadFieldInfo:
    """
    Metadata for a payload field in a Qdrant ODM model.

    Attributes:
        index:
            Optional payload index type hint such as "keyword", "integer", or "datetime".
        alias:
            Optional alternate payload key name used during serialization.
    """

    index: PayloadSchemaLiteral | None = None
    alias: str | None = None

    keyword: KeywordIndexOptions | None = None
    integer: IntegerIndexOptions | None = None
    float_: FloatIndexOptions | None = None
    bool_: BoolIndexOptions | None = None
    geo: GeoIndexOptions | None = None
    datetime: DatetimeIndexOptions | None = None
    text: TextIndexOptions | None = None
    uuid: UuidIndexOptions | None = None


@dataclass(slots=True)
class VectorFieldInfo:
    """
    Metadata for a dense named vector field.

    Attributes:
        name:
            The Qdrant named vector key.
        size:
            The vector dimensionality.
        distance:
            The distance metric, for example "Cosine".
        on_disk:
            Optional flag indicating whether the vector should be stored on disk.
    """

    name: str
    size: int
    distance: str
    on_disk: bool | None = None


@dataclass(slots=True)
class SparseVectorFieldInfo:
    """
    Metadata for a sparse named vector field.

    Attributes:
        name:
            The Qdrant sparse vector key.
    """

    name: str


def PayloadField(
    default: Any = ...,
    *,
    index: PayloadSchemaLiteral | None = None,
    alias: str | None = None,
    keyword: KeywordIndexOptions | None = None,
    integer: IntegerIndexOptions | None = None,
    float_: FloatIndexOptions | None = None,
    bool_: BoolIndexOptions | None = None,
    geo: GeoIndexOptions | None = None,
    datetime: DatetimeIndexOptions | None = None,
    text: TextIndexOptions | None = None,
    uuid: UuidIndexOptions | None = None,
):
    """
    Declare a payload field with optional ODM-specific metadata and
    payload-index configuration.
    """
    payload_info = PayloadFieldInfo(
        index=index,
        alias=alias,
        keyword=keyword,
        integer=integer,
        float_=float_,
        bool_=bool_,
        geo=geo,
        datetime=datetime,
        text=text,
        uuid=uuid,
    )
    json_schema_extra = {"qdrant_payload": payload_info}
    return Field(default=default, alias=alias, json_schema_extra=json_schema_extra)


class VectorField:
    """
    Descriptor used to declare a dense named vector on a model class.

    This is not an instance payload field. Instead, it is a schema-level definition
    that tells the ODM how the model maps to a Qdrant dense vector configuration.
    """

    def __init__(
        self,
        *,
        name: str,
        size: int,
        distance: str,
        on_disk: bool | None = None,
    ) -> None:
        """
        Initialize a dense vector field descriptor.

        Args:
            name:
                Qdrant named vector key.
            size:
                Dense vector dimensionality.
            distance:
                Distance metric such as "Cosine".
            on_disk:
                Optional storage hint for Qdrant.
        """
        self.info = VectorFieldInfo(name=name, size=size, distance=distance, on_disk=on_disk)
        self.attr_name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """
        Record the attribute name used on the owning model class.
        """
        self.attr_name = name

    def __get__(self, instance: Any, owner: type | None = None) -> "VectorField":
        """
        Return the descriptor itself.

        The descriptor represents schema metadata, not an instance value.
        """
        return self


class SparseVectorField:
    """
    Descriptor used to declare a sparse named vector on a model class.

    This is a schema-level definition and is not stored as a normal model field value.
    """

    def __init__(self, *, name: str) -> None:
        """
        Initialize a sparse vector field descriptor.

        Args:
            name:
                Qdrant sparse named vector key.
        """
        self.info = SparseVectorFieldInfo(name=name)
        self.attr_name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """
        Record the attribute name used on the owning model class.
        """
        self.attr_name = name

    def __get__(self, instance: Any, owner: type | None = None) -> "SparseVectorField":
        """
        Return the descriptor itself.

        The descriptor represents schema metadata, not an instance value.
        """
        return self