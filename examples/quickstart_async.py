"""Example demonstrating high-concurrency async operations using AsyncClient."""

from __future__ import annotations

import asyncio

from zero_context import AsyncClient, Config


async def fetch_task(client: AsyncClient, task_id: int) -> int:
    resp = await client.get("/get", params={"task_id": task_id})
    return resp.status_code


async def main() -> None:
    config = Config(
        base_url="https://httpbin.org",
        max_connections=50,
        max_keepalive_connections=20,
        http2=True,
    )

    async with AsyncClient(config) as client:
        print("[Async] Dispatching 10 concurrent requests multiplexed over HTTP/2...")
        tasks = [fetch_task(client, i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        print(f"Completed {len(results)} requests. Status codes: {set(results)}")


if __name__ == "__main__":
    asyncio.run(main())
