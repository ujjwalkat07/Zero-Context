"""Core typed data models and message schemas."""

from __future__ import annotations

from typing import Generic, TypeVar

import msgspec

T = TypeVar("T")


class StreamChunk(msgspec.Struct, frozen=True):
    """A single typed chunk from an event or streaming response."""

    id: str | None = None
    event: str = "message"
    data: str = ""
    retry: int | None = None


class RateLimitInfo(msgspec.Struct, frozen=True):
    """Extracted rate limit status from upstream HTTP headers."""

    limit: int | None = None
    remaining: int | None = None
    reset_at: float | None = None


class APIResponse(msgspec.Struct, Generic[T], frozen=True):
    """Standard generic API response wrapper."""

    data: T
    status: str = "success"
    message: str | None = None
