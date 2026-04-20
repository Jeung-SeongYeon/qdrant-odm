from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def chunked_iter(items: Iterable[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
