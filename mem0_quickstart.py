"""Quickstart powered by Zero-Context, Gemini, and Prisma PGvector."""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Directly import Memory from Zero-Context
from zero_context import Memory

load_dotenv()

# Initialize memory (auto-loads GEMINI_API_KEY and DATABASE_URL from .env)
memory = Memory(api_key=os.getenv("GEMINI_API_KEY"))

# Use the same user ID when adding and searching
USER_ID = "user123"

print("[1] Adding conversation memory...")
# Add a memory (chat messages or text format)
messages = [
    {"role": "user", "content": "I'm a vegetarian and allergic to nuts."},
    {"role": "assistant", "content": "Got it! I'll remember your dietary preferences."},
]
ids = memory.add(messages, user_id=USER_ID)
print("  Ingested memory IDs:", ids)

print("\n[2] Searching memories using PGvector (<=> cosine distance)...")
# Direct vector similarity search
results = memory.search("What are my dietary restrictions?", user_id=USER_ID)
print("  Found memories:")
for r in results:
    print(f"  - [Score: {r.score * 100:.1f}%] {r.memory.text}")

print("\n[3] Zero-Context Prompt Assembly (Ready for LLM Prompt Injection)...")
prompt_context = memory.compile_context("Recommend lunch", user_id=USER_ID)
print(prompt_context)
