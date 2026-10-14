"""Exception hierarchy for ZeroContext SDK."""

from __future__ import annotations


class ZeroContextError(Exception):
    """Base exception for all ZeroContext SDK errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class SerializationError(ZeroContextError):
    """Raised when request payload serialization or response decoding fails."""


class TransportError(ZeroContextError):
    """Raised when a network transport failure occurs."""


class ConnectionTimeoutError(TransportError):
    """Raised when a connection or read attempt times out."""


class APIError(ZeroContextError):
    """Raised when the remote API returns an HTTP error status code (4xx or 5xx)."""

    def __init__(
        self,
        message: str,
        status_code: int,
        response_body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.response_body = response_body
        self.headers = headers or {}


class AuthenticationError(APIError):
    """Raised on 401 Unauthorized or 403 Forbidden responses."""


class NotFoundError(APIError):
    """Raised on 404 Not Found responses."""


class RateLimitError(APIError):
    """Raised on 429 Too Many Requests responses."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        retry_after: float | None = None,
        response_body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            response_body=response_body,
            headers=headers,
        )
        self.retry_after = retry_after


class ServerError(APIError):
    """Raised on 5xx Internal Server Error responses."""
