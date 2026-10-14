"""Basic usage example for Zero-Context Memory with Gemini & PGvector."""

from __future__ import annotations

import os

from zero_context import Memory

# 1. Initialize Memory:
# - Embeddings: Uses Google Gemini (text-embedding-004) if GEMINI_API_KEY is set.
# - Vector DB: Uses PostgreSQL + PGvector if DATABASE_URL is set, otherwise in-memory.
gemini_key = os.getenv("GEMINI_API_KEY")
pg_url = os.getenv("DATABASE_URL")

print(f"Gemini API Key configured : {'Yes' if gemini_key else 'No (using offline mock)'}")
print(f"PGvector DB configured    : {'Yes' if pg_url else 'No (using local in-memory)'}\n")

mem = Memory(
    api_key=gemini_key,
    connection_string=pg_url,
)

# 2. Add conversational facts for a specific user
print("Adding memories for user 'alice'...")
mem.add(
    [
        "User prefers Python and Rust for backend development.",
        "User's favorite framework is FastAPI.",
        "User lives in San Francisco and works remotely.",
    ],
    user_id="alice",
)

# 3. Semantic similarity search
query = "What programming languages does the user prefer?"
print(f"\nSearching for: '{query}'")
results = mem.search(query, user_id="alice", limit=2)

for i, r in enumerate(results, 1):
    print(f"  {i}. [{r.score * 100:.1f}% match] {r.memory.text}")

# 4. Compile ready-to-inject LLM system prompt context
print("\nCompiling LLM Prompt Context:")
prompt_block = mem.compile_context("Recommend an architecture stack", user_id="alice")
print(prompt_block)
