"""ZeroContext - High-performance Python SDK engineered for throughput and low-latency."""

from __future__ import annotations

from zero_context._async_client import AsyncClient
from zero_context._client import Client
from zero_context._config import Config
from zero_context._exceptions import (
    APIError,
    AuthenticationError,
    ConnectionTimeoutError,
    NotFoundError,
    RateLimitError,
    SerializationError,
    ServerError,
    TransportError,
    ZeroContextError,
)
from zero_context._models import APIResponse, RateLimitInfo, StreamChunk
from zero_context._serializer import decode_json, decode_json_raw, encode_json
from zero_context._streaming import AsyncStreamResponse, StreamResponse

__version__ = "0.1.0"

__all__ = [
    "APIError",
    "APIResponse",
    "AsyncClient",
    "AsyncStreamResponse",
    "AuthenticationError",
    "Client",
    "Config",
    "ConnectionTimeoutError",
    "NotFoundError",
    "RateLimitError",
    "RateLimitInfo",
    "SerializationError",
    "ServerError",
    "StreamChunk",
    "StreamResponse",
    "TransportError",
    "ZeroContextError",
    "decode_json",
    "decode_json_raw",
    "encode_json",
]
