"""ZeroMemory / Memory engine powered by Gemini embeddings and Prisma ORM PGVector."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import Any

from zero_context.embeddings import (
    EmbeddingProvider,
    GeminiEmbedding,
    MockGeminiEmbedding,
)
from zero_context.models import MemoryConfig, MemoryItem, SearchResult
from zero_context.store import PrismaVectorStore


class ZeroMemory:
    """Universal high-performance memory layer for AI agents.

    Uses Google Gemini (text-embedding-004) for vector generation and Prisma ORM
    with PostgreSQL pgvector for vector search and management.
    """

    def __init__(
        self,
        api_key: str | None = None,
        connection_string: str | None = None,
        embedder: EmbeddingProvider | None = None,
        store: PrismaVectorStore | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self.config = config or MemoryConfig()

        # Initialize Embedder (Gemini or Mock if key not supplied)
        effective_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if embedder is not None:
            self.embedder = embedder
        elif effective_api_key and not effective_api_key.startswith("AIzaSyYour"):
            self.embedder = GeminiEmbedding(
                api_key=effective_api_key, dimension=self.config.dimension
            )
        else:
            self.embedder = MockGeminiEmbedding(dimension=self.config.dimension)

        # Initialize Vector Store (Prisma PGVector)
        self.store = store or PrismaVectorStore(
            connection_string=connection_string or os.getenv("DATABASE_URL"),
            dimension=self.embedder.dimension,
        )

    def _split_into_facts(self, text: str) -> list[str]:
        """Split a text block into individual factual sentences."""
        sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
        facts = [s.strip() for s in sentences if len(s.strip()) > 3]
        return facts if facts else [text.strip()]

    def add(
        self,
        content: str | Sequence[str] | Sequence[dict[str, str]],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Generate Gemini embeddings and ingest records via Prisma ORM."""
        raw_texts: list[str] = []

        if isinstance(content, str):
            raw_texts.extend(self._split_into_facts(content))
        elif isinstance(content, (list, tuple)):
            for item in content:
                if isinstance(item, str):
                    raw_texts.extend(self._split_into_facts(item))
                elif isinstance(item, dict) and "content" in item:
                    role = item.get("role", "message")
                    role_text = f"[{role}]: {item['content']}"
                    raw_texts.extend(self._split_into_facts(role_text))

        if not raw_texts:
            return []

        vectors = self.embedder.embed_batch(raw_texts)
        created_items: list[MemoryItem] = []

        for text, vec in zip(raw_texts, vectors, strict=False):
            item = MemoryItem.create(
                text=text,
                vector=vec,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
            )
            created_items.append(item)

        self.store.add_batch(created_items)
        return [item.id for item in created_items]

    def search(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 5,
        threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform nearest-neighbor vector search in Prisma PGvector using Gemini embeddings."""
        query_vec = self.embedder.embed_text(query)
        return self.store.search(
            query_vector=query_vec,
            top_k=limit,
            user_id=user_id,
            session_id=session_id,
            metadata_filter=metadata_filter,
            score_threshold=threshold,
        )

    def compile_context(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 5,
        threshold: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
        max_chars: int = 1500,
        header: str = "=== RELEVANT CONTEXT FROM MEMORY ===",
    ) -> str:
        """The Zero-Context signature: compiles relevant memories directly

        into a formatted text block ready for prompt injection into an LLM.
        """
        matches = self.search(
            query=query,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            threshold=threshold,
            metadata_filter=metadata_filter,
        )
        if not matches:
            return ""

        lines = [header]
        current_len = len(header) + 1

        for match in matches:
            score_pct = int(match.score * 100)
            line = f"- {match.memory.text} (relevance: {score_pct}%)"
            if current_len + len(line) + 1 > max_chars:
                break
            lines.append(line)
            current_len += len(line) + 1

        return "\n".join(lines)

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a specific memory by its ID."""
        return self.store.get(memory_id)

    def get_all(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        """Retrieve all stored memories, optionally filtered by user and/or session."""
        return self.store.get_all(user_id=user_id, session_id=session_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory item by ID."""
        return self.store.delete(memory_id)

    def clear(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Clear memories (optionally scoped to a specific user and/or session)."""
        return self.store.clear(user_id=user_id, session_id=session_id)

    def count(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Get the total count of stored memories."""
        return len(self.get_all(user_id=user_id, session_id=session_id))


# User-friendly alias
Memory = ZeroMemory
