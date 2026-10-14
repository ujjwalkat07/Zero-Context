"""High-performance asynchronous client with persistent HTTP/2 pooling and multiplexing."""

from __future__ import annotations

import asyncio
from typing import Any, TypeVar

import httpx

from zero_context._config import Config
from zero_context._exceptions import (
    APIError,
    AuthenticationError,
    ConnectionTimeoutError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
)
from zero_context._retry import (
    IDEMPOTENT_METHODS,
    compute_backoff,
    is_retryable_status,
    parse_retry_after,
)
from zero_context._serializer import decode_json, encode_json
from zero_context._streaming import AsyncStreamResponse

T = TypeVar("T")


class AsyncClient:
    """Asynchronous ZeroContext SDK client engineered for non-blocking high concurrency."""

    def __init__(self, config: Config | None = None, **kwargs: Any) -> None:
        self.config = config or Config(**kwargs)

        headers = {
            "Accept": "application/json",
            "User-Agent": "zero-context-py/0.1.0",
            **self.config.headers,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
            keepalive_expiry=self.config.keepalive_expiry,
        )
        timeout = httpx.Timeout(
            timeout=self.config.timeout,
            connect=self.config.connect_timeout,
        )

        self._http = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            limits=limits,
            timeout=timeout,
            http2=self.config.http2,
        )

    def _handle_error_response(self, response: httpx.Response) -> None:
        headers = dict(response.headers)
        body = response.content
        msg = response.reason_phrase or "API Request Failed"

        if response.status_code in (401, 403):
            raise AuthenticationError(msg, response.status_code, body, headers)
        if response.status_code == 404:
            raise NotFoundError(msg, response.status_code, body, headers)
        if response.status_code == 429:
            retry_after = parse_retry_after(headers.get("retry-after"))
            raise RateLimitError(msg, response.status_code, retry_after, body, headers)
        if response.status_code >= 500:
            raise ServerError(msg, response.status_code, body, headers)
        raise APIError(msg, response.status_code, body, headers)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        response_type: type[T] | None = None,
    ) -> T | httpx.Response:
        """Execute an asynchronous HTTP request with automatic retry and type-safe decoding."""
        upper_method = method.upper()
        content = encode_json(json) if json is not None else None
        req_headers = {"Content-Type": "application/json"} if content is not None else {}
        if headers:
            req_headers.update(headers)

        attempts = 0
        max_attempts = self.config.max_retries if upper_method in IDEMPOTENT_METHODS else 1

        while True:
            attempts += 1
            try:
                response = await self._http.request(
                    method=upper_method,
                    url=path,
                    params=params,
                    content=content,
                    headers=req_headers,
                )

                if is_retryable_status(response.status_code) and attempts < max_attempts:
                    retry_after = parse_retry_after(response.headers.get("retry-after"))
                    delay = (
                        retry_after
                        if retry_after is not None
                        else compute_backoff(attempts, self.config.backoff_factor)
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.is_error:
                    self._handle_error_response(response)

                if response_type is not None:
                    return decode_json(response.content, response_type)

                return response

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as err:
                if attempts < max_attempts:
                    delay = compute_backoff(attempts, self.config.backoff_factor)
                    await asyncio.sleep(delay)
                    continue
                if isinstance(err, (httpx.ReadTimeout, httpx.WriteTimeout)):
                    raise ConnectionTimeoutError(str(err)) from err
                raise TransportError(str(err)) from err

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        response_type: type[T] | None = None,
    ) -> T | httpx.Response:
        return await self.request(
            "GET", path, params=params, headers=headers, response_type=response_type
        )

    async def post(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        response_type: type[T] | None = None,
    ) -> T | httpx.Response:
        return await self.request(
            "POST", path, params=params, json=json, headers=headers, response_type=response_type
        )

    async def put(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        response_type: type[T] | None = None,
    ) -> T | httpx.Response:
        return await self.request(
            "PUT", path, params=params, json=json, headers=headers, response_type=response_type
        )

    async def patch(
        self,
        path: str,
        *,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        response_type: type[T] | None = None,
    ) -> T | httpx.Response:
        return await self.request(
            "PATCH", path, params=params, json=json, headers=headers, response_type=response_type
        )

    async def delete(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        response_type: type[T] | None = None,
    ) -> T | httpx.Response:
        return await self.request(
            "DELETE", path, params=params, headers=headers, response_type=response_type
        )

    async def stream(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        headers: dict[str, str] | None = None,
        target_type: type[T] | None = None,
    ) -> AsyncStreamResponse[T]:
        """Open an asynchronous streaming connection."""
        content = encode_json(json) if json is not None else None
        req_headers = {"Content-Type": "application/json"} if content is not None else {}
        if headers:
            req_headers.update(headers)

        req = self._http.build_request(
            method=method.upper(),
            url=path,
            params=params,
            content=content,
            headers=req_headers,
        )
        response = await self._http.send(req, stream=True)
        if response.is_error:
            await response.aread()
            self._handle_error_response(response)

        return AsyncStreamResponse(response, target_type=target_type)

    async def aclose(self) -> None:
        """Close asynchronous connection pool."""
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
