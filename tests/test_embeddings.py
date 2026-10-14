"""Tests for embedding generators."""

from __future__ import annotations

import math

from zero_context import CallableEmbedding, FastHashEmbedding


def test_fasthash_normalization() -> None:
    embedder = FastHashEmbedding(dimension=128)
    vec = embedder.embed_text("High performance Python vector indexing")
    assert len(vec) == 128

    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


def test_fasthash_empty_string() -> None:
    embedder = FastHashEmbedding(dimension=64)
    vec = embedder.embed_text("")
    assert len(vec) == 64
    assert all(x == 0.0 for x in vec)


def test_callable_embedding() -> None:
    def dummy_embed(text: str) -> list[float]:
        return [1.0, 2.0, 3.0]

    embedder = CallableEmbedding(dummy_embed, dimension=3)
    vec = embedder.embed_text("test")
    assert len(vec) == 3
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-5
