"""Zero-copy streaming iterators for server-sent events and line-delimited streams."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Generic, TypeVar

import httpx

from zero_context._serializer import decode_json

T = TypeVar("T")


class StreamResponse(Generic[T]):
    """Synchronous streaming response wrapper with typed deserialization."""

    def __init__(
        self,
        response: httpx.Response,
        target_type: type[T] | None = None,
    ) -> None:
        self._response = response
        self._target_type = target_type
        self._closed = False

    def __iter__(self) -> Iterator[T | str]:
        try:
            for line in self._response.iter_lines():
                if not line:
                    continue
                if self._target_type is not None:
                    yield decode_json(line.encode("utf-8"), self._target_type)
                else:
                    yield line
        finally:
            self.close()

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        """Iterate over raw bytes chunks directly from the socket."""
        try:
            yield from self._response.iter_bytes(chunk_size=chunk_size)
        finally:
            self.close()

    def close(self) -> None:
        """Close the underlying HTTP transport stream."""
        if not self._closed:
            self._response.close()
            self._closed = True

    def __enter__(self) -> StreamResponse[T]:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class AsyncStreamResponse(Generic[T]):
    """Asynchronous streaming response wrapper with typed deserialization."""

    def __init__(
        self,
        response: httpx.Response,
        target_type: type[T] | None = None,
    ) -> None:
        self._response = response
        self._target_type = target_type
        self._closed = False

    def __aiter__(self) -> AsyncIterator[T | str]:
        return self._iter_lines()

    async def _iter_lines(self) -> AsyncIterator[T | str]:
        try:
            async for line in self._response.aiter_lines():
                if not line:
                    continue
                if self._target_type is not None:
                    yield decode_json(line.encode("utf-8"), self._target_type)
                else:
                    yield line
        finally:
            await self.aclose()

    async def iter_bytes(self, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        """Stream raw byte chunks from the asynchronous socket."""
        try:
            async for chunk in self._response.aiter_bytes(chunk_size=chunk_size):
                yield chunk
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying asynchronous HTTP stream."""
        if not self._closed:
            await self._response.aclose()
            self._closed = True

    async def __aenter__(self) -> AsyncStreamResponse[T]:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
