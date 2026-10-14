"""Fast, lightweight embedding engines for Zero-Context memory."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation engines."""

    dimension: int

    def embed_text(self, text: str) -> list[float]:
        """Convert a text string into a normalized dense embedding vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch encode a list of text strings into normalized vectors."""
        ...


class FastHashEmbedding:
    """Zero-dependency, sub-millisecond local embedding engine.

    Uses character and word n-gram hashing projected onto a unit hypersphere.
    Runs 100% offline with zero external model downloads or API keys.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _hash_token(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], byteorder="little") % self.dimension
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        return idx, sign

    def embed_text(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        cleaned = text.lower().strip()
        if not cleaned:
            return [float(x) for x in vec]

        words = cleaned.split()
        for w in words:
            idx, sign = self._hash_token(w)
            vec[idx] += sign * 1.5

        if len(cleaned) >= 3:
            for i in range(len(cleaned) - 2):
                tri = cleaned[i : i + 3]
                idx, sign = self._hash_token(tri)
                vec[idx] += sign * 0.5

        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return [float(x) for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class CallableEmbedding:
    """Wraps an arbitrary custom embedding function (e.g., from Ollama, ONNX, or OpenAI)."""

    def __init__(self, fn: Callable[[str], list[float]], dimension: int) -> None:
        self._fn = fn
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        vec = self._fn(text)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-12:
            return [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]
