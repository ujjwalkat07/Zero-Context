"""Zero-Context: Memory engine powered by Gemini embeddings and Prisma ORM PGVector."""

from __future__ import annotations

from zero_context.embeddings import (
    CallableEmbedding,
    EmbeddingProvider,
    GeminiEmbedding,
    MockGeminiEmbedding,
)
from zero_context.engine import Memory, ZeroMemory
from zero_context.exceptions import (
    MemoryNotFoundError,
    StorageError,
    ZeroContextError,
)
from zero_context.models import MemoryConfig, MemoryItem, SearchResult
from zero_context.store import PrismaVectorStore

__version__ = "0.4.0"

__all__ = [
    "CallableEmbedding",
    "EmbeddingProvider",
    "GeminiEmbedding",
    "Memory",
    "MemoryConfig",
    "MemoryItem",
    "MemoryNotFoundError",
    "MockGeminiEmbedding",
    "PrismaVectorStore",
    "SearchResult",
    "StorageError",
    "ZeroContextError",
    "ZeroMemory",
]
