# ZeroContext SDK ⚡

A high-performance, asynchronous Python SDK engineered for maximum throughput, low-latency API/RPC communication, and streaming data protocols.

Built from the ground up for modern Python (`>=3.10`), managed with [uv](https://github.com/astral-sh/uv), and powered by `msgspec` and `httpx` with HTTP/2 multiplexing.

---

## 🚀 Performance Highlights

- **Near 10x Faster Serialization**: Uses compiled C-level `msgspec` structs instead of standard library `json` or heavy validation libraries, achieving **~3.7 million ops/sec**.
- **HTTP/2 Multiplexing & Connection Pooling**: Reuses persistent socket connections over HTTP/2, drastically reducing TCP and TLS handshake overhead under high concurrency.
- **Zero-Copy Streaming**: Built-in line and byte generators for consuming Server-Sent Events (SSE) and NDJSON streams without excessive memory allocations.
- **Jittered Exponential Backoff**: Automatic retry policies for idempotent endpoints and transient failures (429, 502, 503, 504) with full jitter to avoid the thundering herd problem.
- **100% Strictly Typed**: Fully compatible with PEP 561 (`py.typed`) and verified with strict `mypy`.

### Serialization Micro-Benchmark (50,000 iterations)

| Operation | `json` (stdlib) | `msgspec` (ZeroContext) | Speedup |
| :--- | :--- | :--- | :--- |
| **Encoding** | ~386,000 ops/s | **~3,727,000 ops/s** | **~9.7x faster** |
| **Decoding** | ~471,000 ops/s | **~1,830,000 ops/s** | **~3.9x faster** |

---

## 📦 Project Structure

```text
Zero-Context/
├── .github/workflows/
│   └── ci.yml                     # Multi-OS & multi-version CI pipeline
├── benchmarks/
│   └── benchmark_serialization.py # High-speed serialization benchmarks
├── examples/
│   ├── quickstart_sync.py         # Sync client usage
│   ├── quickstart_async.py        # High-concurrency async usage
│   └── streaming_example.py       # Chunked streaming consumption
├── src/
│   └── zero_context/
│       ├── __init__.py            # Clean public API and version
│       ├── _async_client.py       # AsyncClient (httpx.AsyncClient + HTTP/2)
│       ├── _client.py             # Client (httpx.Client sync pooling)
│       ├── _config.py             # Config struct (gc=False, frozen=True)
│       ├── _exceptions.py         # Structured exception hierarchy
│       ├── _models.py             # Strongly-typed msgspec Struct models
│       ├── _retry.py              # Full jitter exponential backoff
│       ├── _serializer.py         # Zero-overhead msgspec encoder/decoder
│       ├── _streaming.py          # Low-allocation streaming response handlers
│       └── py.typed               # PEP 561 typing marker
├── tests/
│   ├── conftest.py
│   ├── test_client.py
│   ├── test_async_client.py
│   ├── test_serializer.py
│   ├── test_streaming.py
│   └── test_retry.py
├── pyproject.toml                 # Modern Hatchling configuration with uv
└── README.md
```

---

## 🛠️ Getting Started

### 1. Prerequisites & Installation

Make sure [uv](https://docs.astral.sh/uv/) is installed:

```bash
# Clone the repository
git clone https://github.com/ujjwalkat07/Zero-Context.git
cd Zero-Context

# Synchronize the virtual environment and install dependencies
uv sync
```

---

## 💡 Usage Examples

### 1. Asynchronous Client (Recommended for high concurrency)

```python
import asyncio
from zero_context import AsyncClient, Config

async def main():
    config = Config(
        base_url="https://api.example.com",
        api_key="your-api-key",
        http2=True,
    )

    async with AsyncClient(config) as client:
        # Strongly typed request & response
        response = await client.get("/v1/items")
        print("Status:", response.status_code)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Synchronous Client

```python
from zero_context import Client, Config

config = Config(base_url="https://api.example.com", api_key="your-api-key")

with Client(config) as client:
    response = client.post("/v1/events", json={"event": "login", "user_id": 42})
    print("Response:", response.status_code)
```

### 3. Real-time Streaming

```python
import asyncio
from zero_context import AsyncClient, StreamChunk

async def stream_events():
    async with AsyncClient() as client:
        stream = await client.stream("GET", "/v1/events/live", target_type=StreamChunk)
        async for chunk in stream:
            print(f"Received event {chunk.event}: {chunk.data}")

if __name__ == "__main__":
    asyncio.run(stream_events())
```

---

## 🧪 Development & Quality Assurance

Run code quality checks and tests instantly using `uv`:

```bash
# Run test suite
uv run pytest

# Run strict type checking
uv run mypy

# Lint and format checks
uv run ruff check .
uv run ruff format --check .

# Run performance micro-benchmark
uv run python benchmarks/benchmark_serialization.py
```

---

## 📄 License

MIT
