"""End-to-End Demo with Google Gemini and Prisma ORM PGvector.

To run PostgreSQL with pgvector locally via Docker:
  docker run -d --name zero-pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=zero_context pgvector/pgvector:pg17

Push Prisma Schema to database:
  uv run prisma db push

Usage:
  set GEMINI_API_KEY=your_gemini_api_key
  set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/zero_context
  uv run python examples/pgvector_gemini_demo.py
"""

from __future__ import annotations

import os

from zero_context import GeminiEmbedding, Memory, PrismaVectorStore

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/zero_context"
)


def main() -> None:
    print("=" * 65)
    print("  ZERO-CONTEXT MEMORY - GOOGLE GEMINI + PRISMA PGVECTOR DEMO")
    print("=" * 65)

    if not GEMINI_KEY:
        print("\n[!] Warning: GEMINI_API_KEY environment variable not set.")
        print("    Running in offline demonstration mode with mock embeddings.")
        print("    To use live Gemini: export GEMINI_API_KEY='your_api_key'\n")

    try:
        store = PrismaVectorStore(connection_string=DATABASE_URL, dimension=768)
        # Test connection
        store.get("test-connectivity-probe")
        print(f"[OK] Connected to PostgreSQL via Prisma ORM at: {DATABASE_URL}")
    except Exception as err:
        print(f"[!] Prisma PostgreSQL connection skipped: {err}")
        print("    Falling back to in-memory store for this run.\n")
        store = None

    embedder = GeminiEmbedding(api_key=GEMINI_KEY) if GEMINI_KEY else None
    mem = Memory(embedder=embedder, store=store)

    print("\n[1] Ingesting conversation memories...")
    mem.add(
        [
            "User prefers Python and Go for distributed systems.",
            "User runs PostgreSQL with pgvector managed by Prisma ORM.",
            "User is preparing a high performance SDK college presentation.",
        ],
        user_id="alice",
    )
    print(f"    Stored {mem.count(user_id='alice')} memories for user 'alice'.")

    print("\n[2] Executing Vector Nearest-Neighbor Search (pgvector <=> distance)...")
    results = mem.search("What database and ORM does the user use?", user_id="alice")
    for i, r in enumerate(results, 1):
        print(f"    {i}. [Score: {r.score * 100:.1f}%] {r.memory.text}")

    print("\n[3] Compiling Context for LLM System Prompt...")
    context = mem.compile_context("Recommend an architecture plan", user_id="alice")
    print(context)
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
