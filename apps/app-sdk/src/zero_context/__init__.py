"""Zero-Context: Memory engine powered by Gemini embeddings and Prisma ORM PGVector."""

from pathlib import Path

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

__version__ = "0.2.0"


def get_schema_path() -> Path:
    """Return the filesystem path to the bundled Prisma schema."""
    return Path(__file__).parent / "schema.prisma"


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
    "get_schema_path",
]
