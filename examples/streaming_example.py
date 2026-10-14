"""Example demonstrating zero-copy chunk streaming with AsyncClient."""

from __future__ import annotations

import asyncio

from zero_context import AsyncClient, Config


async def stream_sample() -> None:
    config = Config(base_url="https://httpbin.org", http2=True)

    async with AsyncClient(config) as client:
        print("[Streaming] Streaming line chunks from /stream/5 ...")
        stream = await client.stream("GET", "/stream/5")
        count = 0
        async for line in stream:
            count += 1
            print(f"  Chunk {count}: {line[:80]}...")
        print(f"Finished consuming stream: {count} chunks processed.")


if __name__ == "__main__":
    asyncio.run(stream_sample())
