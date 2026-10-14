"""ZeroMemory / Memory engine - ultra-fast in-process memory layer for AI agents."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from zero_context.embeddings import EmbeddingProvider, FastHashEmbedding
from zero_context.models import MemoryConfig, MemoryItem, SearchResult
from zero_context.store import MemoryStore


class ZeroMemory:
    """Universal high-performance memory layer for AI agents and LLM applications.

    Example:
        >>> from zero_context import ZeroMemory
        >>> mem = ZeroMemory()
        >>> mem.add("User loves Python and building fast tools.", user_id="alice")
        >>> mem.search("What does the user like?", user_id="alice")
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        persist_path: str | Path | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.embedder = embedder or FastHashEmbedding(dimension=self.config.dimension)
        self.store = MemoryStore(dimension=self.embedder.dimension)
        self.persist_path = Path(persist_path) if persist_path else None

        if self.persist_path and self.persist_path.exists():
            self.store.load_from_disk(self.persist_path)

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
        """Ingest conversation text, facts, or chat messages into memory.

        Args:
            content: A single string, list of strings, or chat turns: [{"role": "user", "content": "..."}].
            user_id: Optional user identifier for scoped memory isolation.
            session_id: Optional session/thread identifier.
            metadata: Custom key-value attributes.

        Returns:
            List of generated unique memory record IDs.
        """
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

        if self.config.auto_save and self.persist_path:
            self.store.save_to_disk(self.persist_path)

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
        """Search memory for items semantically relevant to the query string."""
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
        limit: int = 5,
        threshold: float = 0.0,
        max_chars: int = 1500,
        header: str = "=== RELEVANT CONTEXT FROM MEMORY ===",
    ) -> str:
        """The signature Zero-Context feature: compiles relevant memories directly

        into a formatted text block ready for prompt injection into an LLM.
        """
        matches = self.search(
            query=query,
            user_id=user_id,
            limit=limit,
            threshold=threshold,
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

    def get_all(self, user_id: str | None = None) -> list[MemoryItem]:
        """Retrieve all stored memories, optionally filtered by user."""
        return self.store.get_all(user_id=user_id)

    def get(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a specific memory by its ID."""
        return self.store.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory item by ID."""
        success = self.store.delete(memory_id)
        if success and self.config.auto_save and self.persist_path:
            self.store.save_to_disk(self.persist_path)
        return success

    def clear(self, user_id: str | None = None) -> int:
        """Clear memories (optionally scoped to a specific user)."""
        deleted_count = self.store.clear(user_id=user_id)
        if self.config.auto_save and self.persist_path:
            self.store.save_to_disk(self.persist_path)
        return deleted_count

    def count(self, user_id: str | None = None) -> int:
        """Get the total count of stored memories."""
        return len(self.get_all(user_id=user_id))

    def save(self, filepath: str | Path | None = None) -> None:
        """Manually persist memories to disk."""
        dest = Path(filepath) if filepath else self.persist_path
        if not dest:
            raise ValueError("No persistence path specified")
        self.store.save_to_disk(dest)

    def load(self, filepath: str | Path | None = None) -> int:
        """Manually reload memories from disk."""
        src = Path(filepath) if filepath else self.persist_path
        if not src:
            raise ValueError("No persistence path specified")
        return self.store.load_from_disk(src)


# User-friendly alias
Memory = ZeroMemory
