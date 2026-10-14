"""Example demonstrating high-performance synchronous SDK usage."""

from __future__ import annotations

import msgspec

from zero_context import Client, Config


class Account(msgspec.Struct, frozen=True):
    id: str
    email: str
    tier: str = "enterprise"


def main() -> None:
    # Initialize client with persistent connection pool
    config = Config(
        base_url="https://httpbin.org",
        api_key="sk-live-example-token",
        timeout=10.0,
        http2=True,
    )

    with Client(config) as client:
        print("[1] Performing GET request...")
        resp = client.get("/get", params={"user": "developer"})
        print(f"Status Code: {resp.status_code}")

        print("[2] Performing POST request with JSON payload...")
        post_resp = client.post("/post", json={"account_id": "acc_0123", "action": "sync"})
        print(f"Response size: {len(post_resp.content)} bytes")


if __name__ == "__main__":
    main()
