"""Tests for asynchronous AsyncClient."""

from __future__ import annotations

import httpx
import msgspec
import pytest

from zero_context import (
    AsyncClient,
    Config,
    ServerError,
)


class AsyncRecord(msgspec.Struct, frozen=True):
    uuid: str
    processed: bool


@pytest.mark.asyncio
async def test_async_client_get() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/async/status"
        return httpx.Response(200, json={"uuid": "test-uuid-99", "processed": True})

    config = Config(base_url="https://api.test", http2=False)
    client = AsyncClient(config)
    client._http = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )

    try:
        record = await client.get("/v1/async/status", response_type=AsyncRecord)
        assert isinstance(record, AsyncRecord)
        assert record.uuid == "test-uuid-99"
        assert record.processed is True
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_client_post_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(200, json={"uuid": "created-1", "processed": False})

    config = Config(base_url="https://api.test", http2=False)
    client = AsyncClient(config)
    client._http = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )

    try:
        res = await client.post(
            "/v1/async/jobs", json={"task": "process_dataset"}, response_type=AsyncRecord
        )
        assert res.uuid == "created-1"
        assert res.processed is False
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_async_client_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "Database down"})

    config = Config(base_url="https://api.test", max_retries=1, http2=False)
    client = AsyncClient(config)
    client._http = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )

    try:
        with pytest.raises(ServerError) as exc_info:
            await client.get("/v1/crash")
        assert exc_info.value.status_code == 500
    finally:
        await client.aclose()
