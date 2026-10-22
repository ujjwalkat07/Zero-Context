"""Seed script to populate the Zero-Context PostgreSQL database with dummy memory data.

Run with:
    python seed.py
or via uv:
    uv run python seed.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure we can import zero_context
sdk_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(sdk_dir / "src"))

# Load environment
env_path = sdk_dir / ".env"
load_dotenv(dotenv_path=env_path)

from zero_context import Memory  # noqa: E402

DUMMY_RECORDS = [
    # --------------------------------------------------------------------------
    # Customer 101 (Default user for customer_agent.py)
    # --------------------------------------------------------------------------
    {
        "user_id": "customer_101",
        "session_id": "sess_support_001",
        "metadata": {"category": "preference", "priority": "high", "type": "shipping"},
        "facts": [
            "Customer prefers express courier delivery (DHL or FedEx) over standard postal mail.",
            "Customer's default shipping address is 742 Evergreen Terrace, Springfield, OR 97477.",
            "Customer prefers communication via email rather than phone calls.",
        ],
    },
    {
        "user_id": "customer_101",
        "session_id": "sess_support_002",
        "metadata": {"category": "order_history", "order_id": "ORD-8921", "status": "resolved"},
        "facts": [
            "Customer purchased a Keychron Q1 Pro Mechanical Keyboard on Order #ORD-8921.",
            "Customer reported that Order #ORD-8921 had a sticky spacebar switch.",
            "A replacement switch kit was shipped to the customer under warranty tracking #TRK-99214.",
            "Customer has an active 15% discount loyalty coupon code 'LOYALTY15' valid through 2026.",
        ],
    },
    {
        "user_id": "customer_101",
        "session_id": "sess_support_003",
        "metadata": {"category": "tech_profile", "role": "software_engineer"},
        "facts": [
            "Customer is a senior software engineer who uses Python, Neovim, and Linux.",
            "Customer prefers technical explanations and code snippets when troubleshooting.",
            "Customer requested all tax invoices be issued with VAT ID US-987654321.",
        ],
    },
    # --------------------------------------------------------------------------
    # User Alice (Developer user from SDK Documentation)
    # --------------------------------------------------------------------------
    {
        "user_id": "user_alice",
        "session_id": "sess_dev_001",
        "metadata": {"category": "engineering", "level": "senior"},
        "facts": [
            "User prefers Python, Rust, and TypeScript for backend microservice development.",
            "User's favorite web frameworks are FastAPI and Next.js.",
            "User lives in San Francisco, California and works remotely as a Principal AI Engineer.",
            "User is building Zero-Context: a high-performance vector memory layer for AI agents.",
            "User deploys containerized applications with Docker to Kubernetes clusters on GCP.",
            "User enforces strict type checking with Mypy and rapid linting using Ruff.",
        ],
    },
    # --------------------------------------------------------------------------
    # User Bob (Enterprise Product Manager)
    # --------------------------------------------------------------------------
    {
        "user_id": "user_bob",
        "session_id": "sess_prod_001",
        "metadata": {"category": "management", "role": "vp_product"},
        "facts": [
            "User is VP of Product at FinTech Global Corp overseeing four engineering pods.",
            "User prefers concise executive summaries with bullet points and quantifiable metrics.",
            "User tracks quarterly roadmaps in Jira and engineering bugs in Linear.",
            "User's preferred meeting hours are between 9:00 AM and 11:30 AM EST.",
            "User is preparing for the annual SOC2 Type II security and compliance audit.",
        ],
    },
    # --------------------------------------------------------------------------
    # Customer 102 (Retail / Home Office Customer)
    # --------------------------------------------------------------------------
    {
        "user_id": "customer_102",
        "session_id": "sess_retail_001",
        "metadata": {"category": "retail", "vip_level": "gold"},
        "facts": [
            "Customer purchased an Ergohuman Elite Ergonomic Chair on Order #ORD-6612.",
            "Customer requested weekend-only deliveries due to office working hours.",
            "Customer prefers eco-friendly recyclable packaging for all orders.",
            "Customer registered for the 3-year extended hardware warranty program.",
        ],
    },
    # --------------------------------------------------------------------------
    # Knowledge Base / Company Policies
    # --------------------------------------------------------------------------
    {
        "user_id": "kb_system",
        "session_id": "sess_policy_001",
        "metadata": {"category": "company_policy", "scope": "global"},
        "facts": [
            "Zero-Context return policy offers 30-day hassle-free returns with prepaid shipping labels.",
            "Standard shipping takes 3 to 5 business days and is completely free on orders over $50.",
            "Express overnight delivery is available for $14.99 on orders placed before 2:00 PM EST.",
            "All hardware products include a 2-year manufacturer warranty covering defect repairs and replacements.",
            "International shipping is available to over 80 countries with prepaid customs and duty fees.",
        ],
    },
]


def seed_database() -> None:
    print("=" * 60)
    print("       ZERO-CONTEXT DATABASE SEEDER (PGVECTOR + GEMINI)")
    print("=" * 60)

    db_url = os.getenv("DATABASE_URL")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not db_url:
        print("[ERROR] DATABASE_URL is not set in apps/app-sdk/.env")
        sys.exit(1)
    if not gemini_key:
        print("[ERROR] GEMINI_API_KEY is not set in apps/app-sdk/.env")
        sys.exit(1)

    print(f"Connecting to database at: {db_url.split('@')[-1] if '@' in db_url else db_url[:30]}...")
    memory = Memory(api_key=gemini_key, connection_string=db_url)

    total_inserted = 0
    records_by_user: dict[str, int] = {}

    for group in DUMMY_RECORDS:
        user_id = group["user_id"]
        session_id = group.get("session_id")
        metadata = group.get("metadata", {})
        facts: list[str] = group["facts"]  # type: ignore[assignment]

        print(f"\n[+] Ingesting {len(facts)} memories for user '{user_id}'...")
        ids = memory.add(
            content=facts,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )
        print(f"    Inserted {len(ids)} vector memories.")
        total_inserted += len(ids)
        records_by_user[user_id] = records_by_user.get(user_id, 0) + len(ids)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Seeding Complete! Total records inserted: {total_inserted}")
    print("Breakdown by user:")
    for uid, count in records_by_user.items():
        print(f"  * {uid:15}: {count} memories")
    print("=" * 60)

    # Verification: Perform test semantic search
    print("\n[VERIFY] Testing semantic search on newly seeded data...")
    test_query = "What keyboard did the customer order and what issue did they have?"
    print(f"Query: '{test_query}' (user_id='customer_101')")
    results = memory.search(query=test_query, user_id="customer_101", limit=3)
    for idx, r in enumerate(results, 1):
        print(f"  {idx}. [{r.score * 100:.1f}% Match] {r.memory.text}")

    print("\n[DONE] Database is now successfully seeded and verified!")


if __name__ == "__main__":
    seed_database()
