"""Tests for PrismaVectorStore."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from zero_context import MemoryItem, PrismaVectorStore
from zero_context.exceptions import StorageError


def test_prisma_store_config() -> None:
    store = PrismaVectorStore(connection_string="postgresql://localhost:5432/test_db", dimension=768)
    assert store.dimension == 768
    assert "localhost:5432/test_db" in store.connection_string


def test_prisma_store_error_handling() -> None:
    store = PrismaVectorStore(connection_string="postgresql://localhost:5432/test")
    store.aconnect = AsyncMock(side_effect=StorageError("Failed to connect to PostgreSQL via Prisma"))

    with pytest.raises(StorageError) as exc_info:
        store.search(query_vector=[0.1] * 768)
    assert "Failed to connect to PostgreSQL via Prisma" in str(exc_info.value)


def test_prisma_store_search_parsing() -> None:
    async def _test() -> None:
        store = PrismaVectorStore(connection_string="postgresql://localhost:5432/test")
        store.aconnect = AsyncMock()

        mock_prisma = MagicMock()
        mock_row = {
            "id": "mem_001",
            "text": "User likes Python and Prisma ORM",
            "user_id": "alice",
            "session_id": "session_1",
            "metadata": {"role": "developer"},
            "created_at": 1700000000.0,
            "updated_at": 1700000000.0,
            "score": 0.92,
        }
        mock_prisma.query_raw = AsyncMock(return_value=[mock_row])
        store._prisma = mock_prisma

        results = await store.asearch(query_vector=[0.1] * 768, top_k=1, user_id="alice")
        assert len(results) == 1
        assert results[0].memory.id == "mem_001"
        assert results[0].memory.text == "User likes Python and Prisma ORM"
        assert results[0].score == 0.92

    asyncio.run(_test())


def test_prisma_store_add_batch_executes_raw() -> None:
    async def _test() -> None:
        store = PrismaVectorStore(connection_string="postgresql://localhost:5432/test")
        store.aconnect = AsyncMock()

        mock_prisma = MagicMock()
        mock_prisma.execute_raw = AsyncMock()
        store._prisma = mock_prisma

        item = MemoryItem.create("Test item", vector=[0.1] * 768, user_id="alice")
        await store.aadd_batch([item])

        assert mock_prisma.execute_raw.called

    asyncio.run(_test())
