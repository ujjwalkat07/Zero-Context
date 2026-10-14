"""Test fixtures and mock helpers."""

from __future__ import annotations

import pytest

from zero_context import Config


@pytest.fixture
def test_config() -> Config:
    return Config(
        base_url="https://api.test.local",
        api_key="sk-test-secret-key-12345",
        timeout=5.0,
        connect_timeout=2.0,
        max_retries=2,
        backoff_factor=0.01,
        http2=False,  # disabled for simple MockTransport testing
    )
