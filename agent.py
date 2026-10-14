"""Agent with Long-Term Vector Memory using Zero-Context, Gemini, and Prisma PGvector.

This agent demonstrates:
1. Long-term memory ingestion & storage in PostgreSQL (PGvector via Prisma ORM)
2. Semantic similarity vector retrieval using Google Gemini (text-embedding-004)
3. Zero-Context dynamic prompt injection (< 1ms context assembly)
4. Context-aware conversational responses via Gemini LLM
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

from zero_context import Memory

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")


class MemoryAgent:
    """An AI Agent equipped with long-term semantic memory powered by Zero-Context."""

    def __init__(
        self,
        name: str = "Aria",
        user_id: str = "user_default",
        llm_model: str = "gemini-3.5-flash-lite",
    ) -> None:
        self.name = name
        self.user_id = user_id
        self.llm_model = llm_model

        # 1. Initialize Zero-Context Memory Engine (Gemini Embeddings + Prisma PGvector)
        self.memory = Memory(
            api_key=GEMINI_API_KEY,
            connection_string=DATABASE_URL,
        )

        # 2. Initialize Gemini LLM Client
        self.ai = genai.Client(api_key=GEMINI_API_KEY)

    def remember(self, fact_or_conversation: str) -> list[str]:
        """Ingest facts directly into the PostgreSQL PGvector database."""
        ids = self.memory.add(fact_or_conversation, user_id=self.user_id)
        return ids

    def recall(self, query: str, limit: int = 3) -> list[dict[str, object]]:
        """Perform vector similarity search (<=> cosine distance) in PostgreSQL."""
        results = self.memory.search(query, user_id=self.user_id, limit=limit)
        return [
            {
                "id": r.memory.id,
                "text": r.memory.text,
                "relevance": round(r.score * 100, 1),
            }
            for r in results
        ]

    def show_memories(self) -> None:
        """List all persistent memories for this user stored in the database."""
        all_memories = self.memory.get_all(user_id=self.user_id)
        print(f"\n[Database] Total memories for '{self.user_id}': {len(all_memories)}")
        for i, m in enumerate(all_memories, 1):
            print(f"  {i}. [ID: {m.id}] {m.text}")

    def chat(self, user_message: str) -> str:
        """Context-augmented conversational turn:

        1. Vector search in PGvector to retrieve pertinent memories.
        2. Compile relevant context block into the system prompt.
        3. Generate response using Gemini LLM.
        4. Ingest user turn into PGvector so the agent remembers future details.
        """
        # Step 1: Sub-millisecond context retrieval from PGvector
        context_block = self.memory.compile_context(
            query=user_message,
            user_id=self.user_id,
            limit=4,
            threshold=0.1,
            header="=== USER LONG-TERM MEMORY CONTEXT ===",
        )

        system_instruction = (
            f"You are {self.name}, an intelligent personal AI assistant.\n"
            f"You have access to persistent vector memory about the user.\n"
            f"Use the relevant context below to personalize your response concisely.\n\n"
            f"{context_block}\n"
        )

        # Step 2: Generate personalized response using Gemini LLM
        prompt = f"{system_instruction}\nUser: {user_message}\n{self.name}:"
        response = self.ai.models.generate_content(
            model=self.llm_model,
            contents=prompt,
        )
        reply = response.text.strip() if response.text else "I am here to help."

        # Step 3: Persist new facts into PostgreSQL PGvector
        self.memory.add(user_message, user_id=self.user_id)

        return reply


def run_demo() -> None:
    print("=" * 68)
    print("  ZERO-CONTEXT MEMORY AGENT (GEMINI + PRISMA PGVECTOR)")
    print("=" * 68)

    # Initialize agent for user 'alex'
    agent = MemoryAgent(name="Aria", user_id="alex")

    print("\n[Step 1] Ingesting initial profile facts into PGvector database...")
    agent.remember("I am a software engineer specializing in Python, Rust, and distributed systems.")
    agent.remember("I live in Seattle, and I am training for a half-marathon.")
    agent.remember("I am severely allergic to shellfish and peanuts.")
    agent.remember("My favorite tech stack is FastAPI with PostgreSQL and Redis.")
    print("  + Profile facts embedded with text-embedding-004 and saved to PGvector!")

    # Show database records
    agent.show_memories()

    print("\n[Step 2] Vector Search Demonstration (Raw PGvector Cosine Distance)...")
    query = "What programming languages and frameworks do I use?"
    matches = agent.recall(query, limit=2)
    print(f"  Query: '{query}'")
    for m in matches:
        print(f"    - [{m['relevance']}% similarity] {m['text']}")

    print("\n[Step 3] Conversational Memory Augmentation...")

    # Conversation Turn 1: Asking for a recipe recommendation
    print("\nTurn 1:")
    user_prompt_1 = "Can you suggest 3 quick dinner ideas for me tonight?"
    print(f"User: {user_prompt_1}")
    response_1 = agent.chat(user_prompt_1)
    print(f"\nAria: {response_1}\n")

    # Conversation Turn 2: Asking for a weekend tech project recommendation
    print("Turn 2:")
    user_prompt_2 = "What kind of personal side project would you suggest I build this weekend?"
    print(f"User: {user_prompt_2}")
    response_2 = agent.chat(user_prompt_2)
    print(f"\nAria: {response_2}\n")

    print("=" * 68)
    print("  Agent demonstration completed successfully!")
    print("=" * 68)


if __name__ == "__main__":
    run_demo()
