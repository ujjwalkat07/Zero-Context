"""Core data models and schema for Zero-Context memory items."""

from __future__ import annotations

import time
import uuid
from typing import Any

import msgspec


class MemoryItem(msgspec.Struct, frozen=True):
    """An individual immutable memory record."""

    id: str
    text: str
    vector: list[float]
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)
    created_at: float = msgspec.field(default_factory=time.time)
    updated_at: float = msgspec.field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        text: str,
        vector: list[float],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        memory_id: str | None = None,
    ) -> MemoryItem:
        now = time.time()
        return cls(
            id=memory_id or str(uuid.uuid4())[:8],
            text=text,
            vector=vector,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )


class SearchResult(msgspec.Struct, frozen=True):
    """Ranked memory match with computed cosine similarity score."""

    memory: MemoryItem
    score: float


class MemoryConfig(msgspec.Struct, frozen=True, gc=False):
    """Configuration options for Zero-Context memory engine."""

    dimension: int = 256
    similarity_threshold: float = 0.0
    default_top_k: int = 5
    max_context_chars: int = 1500
    auto_save: bool = True
