from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterator[list[T]]:
    """
    Yield fixed-size chunks from a sequence.

    This helper slices the input sequence into lists of at most `size` elements,
    preserving the original order.

    Args:
        items:
            The input sequence to split.
        size:
            The maximum number of items per chunk. Must be at least 1.

    Yields:
        Lists containing up to `size` items from the input sequence.

    Raises:
        ValueError:
            If `size` is less than 1.
    """
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def chunked_iter(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """
    Yield fixed-size chunks from a general iterable.

    Unlike `chunked`, this helper does not require random access or a known length.
    It incrementally accumulates items into batches and yields them as soon as the
    batch reaches the requested size.

    Args:
        items:
            The input iterable to split.
        size:
            The maximum number of items per chunk. Must be at least 1.

    Yields:
        Lists containing up to `size` items from the input iterable.

    Raises:
        ValueError:
            If `size` is less than 1.
    """
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