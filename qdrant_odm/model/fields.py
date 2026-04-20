from dataclasses import dataclass
from typing import Any

from pydantic import Field


@dataclass(slots=True)
class PayloadFieldInfo:
    index: str | None = None
    alias: str | None = None


@dataclass(slots=True)
class VectorFieldInfo:
    name: str
    size: int
    distance: str
    on_disk: bool | None = None


@dataclass(slots=True)
class SparseVectorFieldInfo:
    name: str


def PayloadField(
    default: Any = ...,
    *,
    index: str | None = None,
    alias: str | None = None,
):
    payload_info = PayloadFieldInfo(index=index, alias=alias)
    json_schema_extra = {"qdrant_payload": payload_info}
    return Field(default=default, alias=alias, json_schema_extra=json_schema_extra)


class VectorField:
    def __init__(
        self,
        *,
        name: str,
        size: int,
        distance: str,
        on_disk: bool | None = None,
    ) -> None:
        self.info = VectorFieldInfo(name=name, size=size, distance=distance, on_disk=on_disk)
        self.attr_name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.attr_name = name

    def __get__(self, instance: Any, owner: type | None = None) -> "VectorField":
        return self


class SparseVectorField:
    def __init__(self, *, name: str) -> None:
        self.info = SparseVectorFieldInfo(name=name)
        self.attr_name: str | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.attr_name = name

    def __get__(self, instance: Any, owner: type | None = None) -> "SparseVectorField":
        return self
