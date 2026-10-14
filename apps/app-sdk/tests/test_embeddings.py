"""Tests for Gemini and mock embeddings."""

from __future__ import annotations

import math

from zero_context import CallableEmbedding, MockGeminiEmbedding


def test_mock_gemini_normalization() -> None:
    embedder = MockGeminiEmbedding(dimension=768)
    vec = embedder.embed_text("Testing Google Gemini text-embedding-004")
    assert len(vec) == 768

    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-4


def test_mock_gemini_empty_string() -> None:
    embedder = MockGeminiEmbedding(dimension=768)
    vec = embedder.embed_text("")
    assert len(vec) == 768
    assert all(x == 0.0 for x in vec)


def test_callable_embedding() -> None:
    def custom_fn(text: str) -> list[float]:
        return [0.1] * 768

    embedder = CallableEmbedding(custom_fn, dimension=768)
    vec = embedder.embed_text("test")
    assert len(vec) == 768
    assert vec[0] == 0.1
