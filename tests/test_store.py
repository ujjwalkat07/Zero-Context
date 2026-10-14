"""Tests for vector store operations and cosine similarity math."""

from __future__ import annotations

from pathlib import Path

from zero_context import MemoryItem, MemoryStore


def test_store_vector_search() -> None:
    store = MemoryStore(dimension=3)

    # Unit vectors
    item1 = MemoryItem.create("Item X", vector=[1.0, 0.0, 0.0], user_id="u1")
    item2 = MemoryItem.create("Item Y", vector=[0.0, 1.0, 0.0], user_id="u1")
    item3 = MemoryItem.create("Item Z", vector=[0.0, 0.0, 1.0], user_id="u2")

    store.add_batch([item1, item2, item3])
    assert len(store) == 3

    # Query matches Item X perfectly
    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].memory.text == "Item X"
    assert abs(results[0].score - 1.0) < 1e-5


def test_store_metadata_filtering() -> None:
    store = MemoryStore(dimension=2)

    item1 = MemoryItem.create(
        "Doc A", vector=[1.0, 0.0], metadata={"category": "tech", "priority": "high"}
    )
    item2 = MemoryItem.create(
        "Doc B", vector=[1.0, 0.0], metadata={"category": "finance", "priority": "low"}
    )

    store.add_batch([item1, item2])

    filtered = store.search(
        query_vector=[1.0, 0.0],
        metadata_filter={"category": "finance"},
    )
    assert len(filtered) == 1
    assert filtered[0].memory.text == "Doc B"


def test_store_disk_io(tmp_path: Path) -> None:
    filepath = tmp_path / "store.json"
    store1 = MemoryStore(dimension=2)
    store1.add(MemoryItem.create("Persistent Doc", vector=[0.5, 0.5]))
    store1.save_to_disk(filepath)

    store2 = MemoryStore(dimension=2)
    loaded = store2.load_from_disk(filepath)
    assert loaded == 1
    assert len(store2) == 1
    assert store2.get_all()[0].text == "Persistent Doc"
