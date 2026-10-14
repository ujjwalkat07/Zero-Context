"""Micro-benchmark comparing standard json vs msgspec struct serialization."""

from __future__ import annotations

import json
import time

import msgspec

from zero_context import decode_json, encode_json


class BenchmarkPayload(msgspec.Struct, frozen=True):
    id: str
    user_id: int
    score: float
    tags: list[str]
    metadata: dict[str, str]


def run_benchmark(iterations: int = 50_000) -> None:
    payload = BenchmarkPayload(
        id="evt_01HZYX00ABCDEF123456",
        user_id=12345678,
        score=99.987,
        tags=["production", "telemetry", "streaming", "us-east"],
        metadata={"region": "us-east-1", "host": "worker-42.internal"},
    )

    # 1. msgspec encoding
    start = time.perf_counter()
    raw_bytes: bytes = b""
    for _ in range(iterations):
        raw_bytes = encode_json(payload)
    msgspec_encode_dur = time.perf_counter() - start

    # 2. stdlib json encoding
    py_dict = {
        "id": payload.id,
        "user_id": payload.user_id,
        "score": payload.score,
        "tags": payload.tags,
        "metadata": payload.metadata,
    }
    start = time.perf_counter()
    json_str: str = ""
    for _ in range(iterations):
        json_str = json.dumps(py_dict)
    json_encode_dur = time.perf_counter() - start

    # 3. msgspec decoding into typed struct
    start = time.perf_counter()
    for _ in range(iterations):
        _ = decode_json(raw_bytes, BenchmarkPayload)
    msgspec_decode_dur = time.perf_counter() - start

    # 4. stdlib json decoding into dict
    start = time.perf_counter()
    for _ in range(iterations):
        _ = json.loads(json_str)
    json_decode_dur = time.perf_counter() - start

    print("==================================================")
    print(f"BENCHMARK RESULTS ({iterations:,} iterations):")
    print("==================================================")
    print(
        f"Encode (json.dumps) : {json_encode_dur:.4f}s ({iterations / json_encode_dur:,.0f} ops/s)"
    )
    print(
        f"Encode (msgspec)    : {msgspec_encode_dur:.4f}s ({iterations / msgspec_encode_dur:,.0f} ops/s)"
    )
    print(f"  -> Speedup: {json_encode_dur / msgspec_encode_dur:.2f}x faster\n")
    print(
        f"Decode (json.loads) : {json_decode_dur:.4f}s ({iterations / json_decode_dur:,.0f} ops/s)"
    )
    print(
        f"Decode (msgspec)    : {msgspec_decode_dur:.4f}s ({iterations / msgspec_decode_dur:,.0f} ops/s)"
    )
    print(f"  -> Speedup: {json_decode_dur / msgspec_decode_dur:.2f}x faster")
    print("==================================================")


if __name__ == "__main__":
    run_benchmark()
