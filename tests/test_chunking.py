from qdrant_odm.utils.chunking import chunked


def test_chunked_splits_sequence() -> None:
    batches = list(chunked([1, 2, 3, 4, 5], 2))
    assert batches == [[1, 2], [3, 4], [5]]
