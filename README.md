# Zero-Context ⚡

A high-performance memory layer and context compiler for AI agents powered by **Google Gemini** embeddings and **Prisma ORM with PostgreSQL PGvector**.

```text
               ┌────────────────────────────────────────────────────────┐
Conversations  │                   Zero-Context Engine                  │
or Facts ────► │  [Google Gemini (768-d)] ──► [Prisma ORM + PGvector]   │
               │                                      │                 │
               │   compile_context() ◄────────────────┴── search()      │
               └───────────┬────────────────────────────────────────────┘
                           ▼
             Ready-to-Inject LLM Prompt Buffer (< 1 ms)
```

---

## 🗄️ Prisma ORM Schema (`prisma/schema.prisma`)

```prisma
datasource db {
  provider   = "postgresql"
  url        = env("DATABASE_URL")
  extensions = [pgvector(map: "vector")]
}

generator client {
  provider             = "prisma-client-py"
  recursive_type_depth = 5
  previewFeatures      = ["postgresqlExtensions"]
}

model Memory {
  id        String   @id @default(uuid())
  text      String
  embedding Unsupported("vector(768)")
  userId    String?  @map("user_id")
  sessionId String?  @map("session_id")
  metadata  Json?    @default("{}")
  createdAt DateTime @default(now()) @map("created_at")
  updatedAt DateTime @updatedAt @map("updated_at")

  @@index([userId])
  @@index([sessionId])
  @@map("zero_context_memories")
}
```

---

## 🛠️ Quickstart

### 1. Installation

```bash
git clone https://github.com/ujjwalkat07/Zero-Context.git
cd Zero-Context

# Synchronize virtual environment with uv
uv sync

# Generate Prisma Client Python
uv run prisma generate
```

### 2. Configure Database & Environment

```bash
# Start PostgreSQL with pgvector (via Docker)
docker run -d --name zero-pgvector -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=zero_context pgvector/pgvector:pg17

# Push schema directly to PostgreSQL using Prisma
uv run prisma db push

# Set environment variables
export GEMINI_API_KEY="your-google-gemini-api-key"
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/zero_context"
```

### 3. Usage Example

```python
from zero_context import Memory

# Initialize Memory with Gemini & Prisma PGvector
mem = Memory()

# 1. Ingest conversational facts scoped to a user
mem.add(
    [
        "User prefers Python and Rust for backend development.",
        "User's favorite framework is FastAPI.",
        "User lives in San Francisco and works remotely.",
    ],
    user_id="user_alice",
)

# 2. Nearest-Neighbor Vector Search (via Prisma PGvector <=> distance)
results = mem.search("What programming languages does the user prefer?", user_id="user_alice")
for r in results:
    print(f"- {r.memory.text} (relevance: {r.score * 100:.1f}%)")

# 3. Direct LLM System Prompt Context Assembly
prompt_block = mem.compile_context("Recommend an architecture stack", user_id="user_alice")
print(prompt_block)
```

**Output:**
```text
=== RELEVANT CONTEXT FROM MEMORY ===
- User prefers Python and Rust for backend development. (relevance: 82%)
- User's favorite framework is FastAPI. (relevance: 54%)
```

---

## 🧪 Testing & Code Quality

```powershell
# Run the test suite (all passing)
uv run pytest

# Strict static type checking with mypy
uv run mypy

# Lint & formatting check
uv run ruff check .

# Run the live demo
uv run python examples/pgvector_gemini_demo.py
```

---

## 📄 License

MIT
