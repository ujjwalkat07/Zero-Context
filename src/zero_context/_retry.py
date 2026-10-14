"""High-performance retry logic with jittered exponential backoff."""

from __future__ import annotations

import random
from typing import Final

RETRY_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
IDEMPOTENT_METHODS: Final[frozenset[str]] = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})


def compute_backoff(attempt: int, backoff_factor: float, max_backoff: float = 60.0) -> float:
    """Compute jittered exponential backoff delay in seconds."""
    # Full jitter: random sleep between 0 and min(max_backoff, factor * 2^(attempt))
    ceiling = min(max_backoff, backoff_factor * (2.0**attempt))
    return random.uniform(0.0, ceiling)


def parse_retry_after(header_value: str | None) -> float | None:
    """Extract seconds from a Retry-After header string if present."""
    if not header_value:
        return None
    try:
        seconds = float(header_value.strip())
        return max(0.0, seconds)
    except (ValueError, TypeError):
        return None


def is_retryable_status(status_code: int) -> bool:
    """Determine whether an HTTP status code qualifies for retry."""
    return status_code in RETRY_STATUS_CODES
