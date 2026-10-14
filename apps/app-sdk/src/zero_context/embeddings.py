"""Gemini embedding engine for Zero-Context memory."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Protocol

from google import genai

from zero_context.exceptions import ZeroContextError


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation engines."""

    dimension: int

    def embed_text(self, text: str) -> list[float]:
        """Convert a text string into a dense embedding vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch encode a list of text strings into dense embedding vectors."""
        ...


class GeminiEmbedding:
    """Production embedding provider powered by Google Gemini (text-embedding-004).

    Produces 768-dimensional embeddings via Google GenAI SDK.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ZeroContextError(
                "Gemini API key is required. Pass api_key or set the GEMINI_API_KEY environment variable."
            )
        self.model = model
        self.dimension = dimension
        self._client = genai.Client(api_key=self.api_key)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string using Gemini."""
        if not text.strip():
            return [0.0] * self.dimension

        response = self._client.models.embed_content(
            model=self.model,
            contents=text,
            config=genai.types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise ZeroContextError("Gemini API returned an empty embedding response.")

        return [float(x) for x in response.embeddings[0].values]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple strings in a single call to Gemini."""
        if not texts:
            return []

        response = self._client.models.embed_content(
            model=self.model,
            contents=list(texts),
            config=genai.types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        if not response.embeddings:
            raise ZeroContextError("Gemini API returned empty batch embeddings.")

        return [
            [float(x) for x in emb.values] if emb.values else [0.0] * self.dimension
            for emb in response.embeddings
        ]


class MockGeminiEmbedding:
    """Offline 768-dimensional deterministic embedding provider.

    Used for unit tests, offline development, or CI environments without requiring
    an external paid Google Gemini API key.
    """

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = dimension

    def _hash_to_vec(self, text: str) -> list[float]:
        cleaned = text.lower().strip()
        if not cleaned:
            return [0.0] * self.dimension

        vec = [0.0] * self.dimension
        for word in cleaned.split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="little") % self.dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign * 1.5

        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 1e-12:
            return [x / norm for x in vec]
        return vec

    def embed_text(self, text: str) -> list[float]:
        return self._hash_to_vec(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vec(t) for t in texts]


class CallableEmbedding:
    """Wraps any user-defined embedding function."""

    def __init__(self, fn: Callable[[str], list[float]], dimension: int = 768) -> None:
        self._fn = fn
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        return [float(x) for x in self._fn(text)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]
