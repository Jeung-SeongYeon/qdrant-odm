from dataclasses import dataclass
from typing import Any

from pydantic import Field


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

    index: str | None = None
    alias: str | None = None


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
    index: str | None = None,
    alias: str | None = None,
):
    """
    Declare a payload field with optional ODM-specific metadata.

    This helper wraps `pydantic.Field(...)` and stores Qdrant ODM metadata
    inside `json_schema_extra["qdrant_payload"]`.

    Args:
        default:
            Default field value passed to Pydantic.
        index:
            Optional payload index type hint for schema generation.
        alias:
            Optional alias used by Pydantic serialization and validation.

    Returns:
        A configured Pydantic field object.
    """
    payload_info = PayloadFieldInfo(index=index, alias=alias)
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