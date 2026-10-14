"""Interactive Terminal Customer Support Agent powered by Zero-Context & Gemini.

This agent demonstrates:
1. Interactive terminal conversation with a customer.
2. Semantic memory search and recall from PostgreSQL (PGvector via Prisma).
3. Dynamic prompt augmentation using compiled memory context.
4. Automatic persistence of customer preferences and conversation history.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from google import genai

from zero_context import Memory

# Load API keys from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY is not set in your .env file.")
    sys.exit(1)


class CustomerSupportAgent:
    """Customer Support AI Assistant with persistent long-term memory."""

    def __init__(
        self,
        customer_id: str = "customer_101",
        model_name: str = "gemini-3.5-flash-lite",
    ) -> None:
        self.customer_id = customer_id
        self.model_name = model_name

        # 1. Initialize Zero-Context Memory SDK
        self.memory = Memory(
            api_key=GEMINI_API_KEY,
            connection_string=DATABASE_URL,
        )

        # 2. Initialize Google Gemini LLM Client
        self.ai = genai.Client(api_key=GEMINI_API_KEY)

    def respond(self, user_query: str) -> str:
        """Process customer query using vector memory + Gemini LLM."""
        # Step A: Semantic retrieval of customer history from PostgreSQL
        context = self.memory.compile_context(
            query=user_query,
            user_id=self.customer_id,
            limit=4,
            threshold=0.2,
            header="=== CUSTOMER PREFERENCES & ORDER HISTORY FROM MEMORY ===",
        )

        # Step B: Assemble context-augmented system prompt
        system_instruction = (
            "You are Nova, an empathetic and professional customer support representative.\n"
            "You have access to persistent long-term memory about this customer.\n"
            "Use the customer background below to personalize your reply concisely.\n\n"
            f"{context}\n"
        )

        full_prompt = f"{system_instruction}\nCustomer: {user_query}\nNova:"

        # Step C: Generate response via Gemini LLM
        response = self.ai.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
        )
        agent_reply = (
            response.text.strip() if response.text else "I am here to assist you."
        )

        # Step D: Save this interaction so future queries remember details
        interaction = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": agent_reply},
        ]
        self.memory.add(interaction, user_id=self.customer_id)

        return agent_reply

    def list_memories(self) -> None:
        """Display all stored facts for this customer in PGvector."""
        items = self.memory.get_all(user_id=self.customer_id)
        print(f"\n--- Stored Memories for [{self.customer_id}] ({len(items)} items) ---")
        if not items:
            print("  (No memories recorded yet.)")
            return
        for idx, item in enumerate(items, 1):
            print(f"  {idx}. [ID: {item.id}] {item.text}")
        print("-------------------------------------------------------")

    def search_memories(self, query: str) -> None:
        """Test vector similarity search directly from terminal."""
        results = self.memory.search(query, user_id=self.customer_id, limit=3)
        print(f"\n--- Vector Search Results for: '{query}' ---")
        if not results:
            print("  (No close matches found.)")
            return
        for r in results:
            print(f"  - [{r.score * 100:.1f}% Match | ID: {r.memory.id}] {r.memory.text}")
        print("---------------------------------------------")

    def clear_memories(self) -> None:
        """Wipe memory records for this customer."""
        count = self.memory.clear(user_id=self.customer_id)
        print(f"\n[Success] Cleared {count} memories for customer '{self.customer_id}'.")


def main() -> None:
    print("=" * 65)
    print("      NOVA CUSTOMER SUPPORT AGENT (POWERED BY ZERO-CONTEXT)")
    print("=" * 65)
    print("Commands:")
    print("  /memories        - View all stored memories for this customer")
    print("  /search <query>  - Test raw vector search (<=> cosine distance)")
    print("  /clear           - Clear this customer's memory database")
    print("  exit / quit      - Exit the application")
    print("=" * 65)

    default_id = "customer_101"
    prompt_id = input(f"Enter Customer ID [default: {default_id}]: ").strip()
    customer_id = prompt_id if prompt_id else default_id

    agent = CustomerSupportAgent(customer_id=customer_id)
    print(f"\n[Agent Online] Connected to PostgreSQL. Ready for customer '{customer_id}'.\n")

    while True:
        try:
            user_input = input(f"{customer_id} > ").strip()
            if not user_input:
                continue

            # Command routing
            if user_input.lower() in ("exit", "quit", "q"):
                print("Thank you for using Nova Customer Support. Goodbye!")
                break

            if user_input.lower() == "/memories":
                agent.list_memories()
                continue

            if user_input.lower().startswith("/search"):
                search_query = user_input[7:].strip()
                if not search_query:
                    print("Usage: /search <your query>")
                else:
                    agent.search_memories(search_query)
                continue

            if user_input.lower() == "/clear":
                agent.clear_memories()
                continue

            # Standard chat interaction
            print("\nNova thinking...")
            reply = agent.respond(user_input)
            print(f"\nNova: {reply}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
