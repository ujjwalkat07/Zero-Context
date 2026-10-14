"""Tests for ZeroMemory engine using PrismaVectorStore and Gemini embeddings."""

from __future__ import annotations

from unittest.mock import MagicMock

from zero_context import MemoryItem, MockGeminiEmbedding, SearchResult, ZeroMemory
from zero_context.store import PrismaVectorStore


def test_split_into_facts() -> None:
    mem = ZeroMemory()
    facts = mem._split_into_facts("User loves Python. User builds fast APIs! User works remotely.")
    assert len(facts) == 3
    assert facts[0] == "User loves Python."
    assert facts[1] == "User builds fast APIs!"
    assert facts[2] == "User works remotely."


def test_compile_context_formatting() -> None:
    mock_store = MagicMock(spec=PrismaVectorStore)
    item1 = MemoryItem.create("User is allergic to peanuts.", vector=[0.1] * 768, user_id="foodie")
    item2 = MemoryItem.create("User dislikes spicy dishes.", vector=[0.1] * 768, user_id="foodie")

    mock_store.search.return_value = [
        SearchResult(memory=item1, score=0.88),
        SearchResult(memory=item2, score=0.45),
    ]

    mem = ZeroMemory(embedder=MockGeminiEmbedding(), store=mock_store)
    ctx = mem.compile_context("peanuts and allergies", user_id="foodie")

    assert "RELEVANT CONTEXT" in ctx
    assert "User is allergic to peanuts." in ctx
    assert "relevance: 88%" in ctx


def test_engine_add_and_crud() -> None:
    mock_store = MagicMock(spec=PrismaVectorStore)
    mock_store.get.return_value = MemoryItem.create("Test item", vector=[0.1] * 768)
    mock_store.delete.return_value = True
    mock_store.clear.return_value = 1
    mock_store.get_all.return_value = [MemoryItem.create("Test item", vector=[0.1] * 768)]

    mem = ZeroMemory(embedder=MockGeminiEmbedding(), store=mock_store)
    ids = mem.add("Test fact for user", user_id="u1")

    assert len(ids) == 1
    assert mock_store.add_batch.called

    # Test get, delete, clear, count
    assert mem.get(ids[0]) is not None
    assert mem.delete(ids[0]) is True
    assert mem.clear(user_id="u1") == 1
    assert mem.count() == 1
