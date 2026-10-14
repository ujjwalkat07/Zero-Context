"""Zero-Context: Ultra-fast, zero-dependency in-process memory engine for AI agents."""

from __future__ import annotations

from zero_context.embeddings import (
    CallableEmbedding,
    EmbeddingProvider,
    FastHashEmbedding,
)
from zero_context.engine import Memory, ZeroMemory
from zero_context.exceptions import (
    MemoryNotFoundError,
    StorageError,
    ZeroContextError,
)
from zero_context.models import MemoryConfig, MemoryItem, SearchResult
from zero_context.store import MemoryStore

__version__ = "0.2.0"

__all__ = [
    "CallableEmbedding",
    "EmbeddingProvider",
    "FastHashEmbedding",
    "Memory",
    "MemoryConfig",
    "MemoryItem",
    "MemoryNotFoundError",
    "MemoryStore",
    "SearchResult",
    "StorageError",
    "ZeroContextError",
    "ZeroMemory",
]
