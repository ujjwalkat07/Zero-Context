"""Retrieval Latency and Throughput Benchmark for Zero-Context Memory."""

from __future__ import annotations

import time

from zero_context import Memory


def run_benchmark() -> None:
    print("=" * 60)
    print("  ZERO-CONTEXT MEMORY - PERFORMANCE BENCHMARK")
    print("=" * 60)

    mem = Memory()

    # Seed with 1,000 distinct memory records
    print("Seeding 1,000 memory items into vector store...")
    items = [
        f"Memory record #{i}: User performed activity in project category_{i % 10}"
        for i in range(1, 1001)
    ]
    t0 = time.perf_counter()
    mem.add(items, user_id="benchmark_user")
    ingest_time = time.perf_counter() - t0
    print(
        f"  Ingested 1,000 items in {ingest_time * 1000:.2f} ms ({len(items) / ingest_time:,.0f} items/s)\n"
    )

    query = "Find user activity in category_5"
    iterations = 5_000

    # Warmup
    for _ in range(100):
        _ = mem.search(query, user_id="benchmark_user", limit=5)

    print(f"Running {iterations:,} vector similarity searches...")
    t_start = time.perf_counter()
    for _ in range(iterations):
        _ = mem.search(query, user_id="benchmark_user", limit=5)
    total_time = time.perf_counter() - t_start

    avg_us = (total_time / iterations) * 1_000_000
    qps = iterations / total_time

    print("-" * 60)
    print(f"  Average Query Latency : {avg_us:.1f} microseconds ({avg_us / 1000:.3f} ms)")
    print(f"  Throughput            : {qps:,.0f} queries/second (QPS)")
    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
