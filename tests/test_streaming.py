"""Tests for sync and async streaming iterators."""

from __future__ import annotations

import httpx
import pytest

from zero_context import AsyncStreamResponse, StreamChunk, StreamResponse


def test_sync_stream_lines() -> None:
    lines = [
        b'{"id":"1","event":"msg","data":"first"}\n',
        b'{"id":"2","event":"msg","data":"second"}\n',
    ]
    response = httpx.Response(200, content=b"".join(lines))
    stream = StreamResponse(response, target_type=StreamChunk)

    chunks = list(stream)
    assert len(chunks) == 2
    assert isinstance(chunks[0], StreamChunk)
    assert chunks[0].id == "1"
    assert chunks[0].data == "first"
    assert chunks[1].id == "2"
    assert chunks[1].data == "second"


@pytest.mark.asyncio
async def test_async_stream_lines() -> None:
    lines = [
        b'{"id":"10","event":"delta","data":"alpha"}\n',
        b'{"id":"20","event":"delta","data":"beta"}\n',
    ]
    response = httpx.Response(200, content=b"".join(lines))
    stream = AsyncStreamResponse(response, target_type=StreamChunk)

    collected: list[StreamChunk] = []
    async for chunk in stream:
        assert isinstance(chunk, StreamChunk)
        collected.append(chunk)

    assert len(collected) == 2
    assert collected[0].id == "10"
    assert collected[1].id == "20"
