"""High-performance vector memory store with vectorized SIMD cosine similarity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from zero_context.exceptions import StorageError
from zero_context.models import MemoryItem, SearchResult


class MemoryStore:
    """In-memory vector store powered by NumPy matrix multiplication and msgspec persistence."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self._items: dict[str, MemoryItem] = {}
        self._id_order: list[str] = []
        self._matrix: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._items)

    def _rebuild_matrix(self) -> None:
        if not self._id_order:
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)
            return
        vectors = [self._items[item_id].vector for item_id in self._id_order]
        self._matrix = np.array(vectors, dtype=np.float32)

    def add(self, item: MemoryItem) -> None:
        """Insert or update a memory record."""
        is_new = item.id not in self._items
        self._items[item.id] = item
        if is_new:
            self._id_order.append(item.id)
            new_vec = np.array([item.vector], dtype=np.float32)
            if self._matrix.shape[0] == 0:
                self._matrix = new_vec
            else:
                self._matrix = np.vstack([self._matrix, new_vec])
        else:
            idx = self._id_order.index(item.id)
            self._matrix[idx] = np.array(item.vector, dtype=np.float32)

    def add_batch(self, items: list[MemoryItem]) -> None:
        """Batch insert memory records efficiently."""
        if not items:
            return
        new_vectors = []
        for item in items:
            if item.id in self._items:
                self.add(item)
            else:
                self._items[item.id] = item
                self._id_order.append(item.id)
                new_vectors.append(item.vector)

        if new_vectors:
            new_mat = np.array(new_vectors, dtype=np.float32)
            if self._matrix.shape[0] == 0:
                self._matrix = new_mat
            else:
                self._matrix = np.vstack([self._matrix, new_mat])

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def get_all(self, user_id: str | None = None) -> list[MemoryItem]:
        if user_id is None:
            return list(self._items.values())
        return [item for item in self._items.values() if item.user_id == user_id]

    def delete(self, memory_id: str) -> bool:
        if memory_id not in self._items:
            return False
        del self._items[memory_id]
        self._id_order.remove(memory_id)
        self._rebuild_matrix()
        return True

    def clear(self, user_id: str | None = None) -> int:
        if user_id is None:
            count = len(self._items)
            self._items.clear()
            self._id_order.clear()
            self._matrix = np.empty((0, self.dimension), dtype=np.float32)
            return count

        to_remove = [item_id for item_id, item in self._items.items() if item.user_id == user_id]
        for item_id in to_remove:
            del self._items[item_id]
            self._id_order.remove(item_id)
        self._rebuild_matrix()
        return len(to_remove)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Compute SIMD-accelerated cosine similarity and return top matches."""
        if len(self._items) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        scores = self._matrix @ q

        results: list[SearchResult] = []
        ranked_indices = np.argsort(-scores)

        for idx in ranked_indices:
            score = float(scores[idx])
            if score < score_threshold:
                break

            item_id = self._id_order[int(idx)]
            item = self._items[item_id]

            if user_id is not None and item.user_id != user_id:
                continue
            if session_id is not None and item.session_id != session_id:
                continue
            if metadata_filter:
                match = all(item.metadata.get(k) == v for k, v in metadata_filter.items())
                if not match:
                    continue

            results.append(SearchResult(memory=item, score=score))
            if len(results) >= top_k:
                break

        return results

    def save_to_disk(self, filepath: str | Path) -> None:
        """Serialize memory records to disk via msgspec."""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            items_list = list(self._items.values())
            data = msgspec.json.encode(items_list)
            path.write_bytes(data)
        except Exception as err:
            raise StorageError(f"Failed to save memory store to '{filepath}': {err}") from err

    def load_from_disk(self, filepath: str | Path) -> int:
        """Load memory records from disk."""
        path = Path(filepath)
        if not path.exists():
            return 0
        try:
            data = path.read_bytes()
            items = msgspec.json.decode(data, type=list[MemoryItem])
            self.clear()
            self.add_batch(items)
            return len(items)
        except Exception as err:
            raise StorageError(f"Failed to load memory store from '{filepath}': {err}") from err
