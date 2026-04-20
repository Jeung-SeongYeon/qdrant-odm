from typing import TypeVar

from qdrant_odm.model.base import QdrantModel

PointIdT = TypeVar("PointIdT", str, int)
QdrantModelT = TypeVar("QdrantModelT", bound=QdrantModel)
