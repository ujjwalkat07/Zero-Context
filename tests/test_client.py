"""Tests for synchronous Client."""

from __future__ import annotations

import httpx
import msgspec
import pytest

from zero_context import (
    AuthenticationError,
    Client,
    Config,
    NotFoundError,
    RateLimitError,
)


class UserProfile(msgspec.Struct, frozen=True):
    id: str
    username: str


def test_client_get_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer sk-test-key"
        assert request.url.path == "/v1/users/42"
        return httpx.Response(200, json={"id": "42", "username": "alice"})

    config = Config(base_url="https://api.test", api_key="sk-test-key", http2=False)
    client = Client(config)
    client._http = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {config.api_key}"},
    )

    profile = client.get("/v1/users/42", response_type=UserProfile)
    assert isinstance(profile, UserProfile)
    assert profile.id == "42"
    assert profile.username == "alice"


def test_client_post_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers.get("Content-Type") == "application/json"
        assert b'"username":"bob"' in request.content
        return httpx.Response(201, json={"id": "99", "username": "bob"})

    config = Config(base_url="https://api.test", http2=False)
    client = Client(config)
    client._http = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )

    res = client.post("/v1/users", json={"username": "bob"}, response_type=UserProfile)
    assert isinstance(res, UserProfile)
    assert res.id == "99"
    assert res.username == "bob"


def test_client_authentication_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Invalid API Key"})

    config = Config(base_url="https://api.test", http2=False)
    client = Client(config)
    client._http = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(AuthenticationError) as exc_info:
        client.get("/v1/secret")
    assert exc_info.value.status_code == 401


def test_client_not_found_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Not Found"})

    config = Config(base_url="https://api.test", http2=False)
    client = Client(config)
    client._http = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(NotFoundError) as exc_info:
        client.get("/v1/missing")
    assert exc_info.value.status_code == 404


def test_client_rate_limit_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "2.5"}, json={"error": "Rate limit exceeded"}
        )

    config = Config(base_url="https://api.test", max_retries=1, http2=False)
    client = Client(config)
    client._http = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RateLimitError) as exc_info:
        client.get("/v1/busy")
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 2.5
