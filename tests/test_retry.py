"""Tests for retry backoff and header parsing logic."""

from __future__ import annotations

from zero_context._retry import (
    compute_backoff,
    is_retryable_status,
    parse_retry_after,
)


def test_compute_backoff_bounds() -> None:
    for attempt in range(1, 5):
        backoff = compute_backoff(attempt, backoff_factor=0.5, max_backoff=10.0)
        assert 0.0 <= backoff <= min(10.0, 0.5 * (2.0**attempt))


def test_parse_retry_after() -> None:
    assert parse_retry_after("12.5") == 12.5
    assert parse_retry_after("0") == 0.0
    assert parse_retry_after("-5") == 0.0
    assert parse_retry_after("invalid-text") is None
    assert parse_retry_after(None) is None


def test_is_retryable_status() -> None:
    assert is_retryable_status(429) is True
    assert is_retryable_status(503) is True
    assert is_retryable_status(502) is True
    assert is_retryable_status(408) is True
    assert is_retryable_status(200) is False
    assert is_retryable_status(400) is False
    assert is_retryable_status(401) is False
    assert is_retryable_status(404) is False
