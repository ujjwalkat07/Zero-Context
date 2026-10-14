"""Tests for ZeroMemory and Memory engine."""

from __future__ import annotations

from pathlib import Path

from zero_context import Memory, ZeroMemory


def test_add_and_search_scoped() -> None:
    mem = ZeroMemory()
    mem.add("Alice loves coding in Python and deploying with Docker.", user_id="alice")
    mem.add("Bob prefers developing in Go with Kubernetes.", user_id="bob")

    alice_res = mem.search("What language does user code in?", user_id="alice")
    assert len(alice_res) > 0
    assert "Python" in alice_res[0].memory.text

    bob_res = mem.search("What language does user code in?", user_id="bob")
    assert len(bob_res) > 0
    assert "Go" in bob_res[0].memory.text


def test_compile_context() -> None:
    mem = ZeroMemory()
    mem.add(
        "User is allergic to peanuts. User dislikes spicy dishes.",
        user_id="foodie",
    )

    ctx = mem.compile_context("peanuts and allergies", user_id="foodie")
    assert "peanuts" in ctx
    assert "RELEVANT CONTEXT" in ctx


def test_crud_operations() -> None:
    mem = ZeroMemory()
    ids = mem.add("Test fact 1", user_id="u1")
    mem.add("Test fact 2", user_id="u2")
    assert mem.count() == 2

    # Get by ID
    item = mem.get(ids[0])
    assert item is not None
    assert item.text == "Test fact 1"

    # Delete
    assert mem.delete(ids[0]) is True
    assert mem.get(ids[0]) is None
    assert mem.count() == 1

    # Clear scoped
    mem.clear(user_id="u2")
    assert mem.count() == 0


def test_persistence(tmp_path: Path) -> None:
    db_file = tmp_path / "memory.zctx"

    mem1 = Memory(persist_path=db_file)
    mem1.add("Postgres is running on port 5432.", user_id="admin")

    # Load in a fresh instance
    mem2 = Memory(persist_path=db_file)
    assert mem2.count() == 1
    res = mem2.search("What port does Postgres use?", user_id="admin")
    assert len(res) == 1
    assert "5432" in res[0].memory.text
