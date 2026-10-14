"""Basic usage example for Zero-Context Memory engine."""

from __future__ import annotations

from zero_context import Memory

# 1. Initialize Memory (Runs 100% locally with zero external dependencies)
mem = Memory(persist_path="./storage/user_memory.zctx")

# 2. Add conversational facts for a specific user
print("Adding memories for user 'alice'...")
mem.add(
    [
        "User prefers Python and Rust for backend development.",
        "User's favorite framework is FastAPI.",
        "User works in European timezone (CET).",
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
