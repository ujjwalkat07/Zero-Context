"""Client configuration engineered for high throughput and connection reuse."""

from __future__ import annotations

import msgspec


class Config(msgspec.Struct, frozen=True, gc=False):
    """Immutable client configuration with optimized socket and pooling defaults.

    Attributes:
        base_url: The root URL for API requests.
        api_key: Optional Bearer authentication token.
        timeout: Total request timeout in seconds.
        connect_timeout: Socket connection establishment timeout in seconds.
        max_retries: Maximum number of retry attempts for idempotent failures.
        backoff_factor: Multiplier for exponential jittered backoff calculation.
        max_connections: Maximum total concurrent sockets in connection pool.
        max_keepalive_connections: Maximum idle persistent sockets in pool.
        keepalive_expiry: Time in seconds to keep idle sockets open.
        http2: Enable HTTP/2 multiplexed streams over single connection.
        headers: Default HTTP headers injected into every request.
    """

    base_url: str = "https://api.zerocontext.dev"
    api_key: str | None = None
    timeout: float = 30.0
    connect_timeout: float = 5.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0
    http2: bool = True
    headers: dict[str, str] = msgspec.field(default_factory=dict)
