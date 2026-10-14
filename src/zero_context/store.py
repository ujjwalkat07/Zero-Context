"""Prisma ORM storage layer for PostgreSQL and PGvector."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from prisma import Prisma
from prisma.types import DatasourceOverride, MemoryWhereInput

from zero_context.exceptions import StorageError
from zero_context.models import MemoryItem, SearchResult

T = TypeVar("T")


_WORKER_LOOP: asyncio.AbstractEventLoop | None = None
_WORKER_THREAD: threading.Thread | None = None
_LOOP_LOCK = threading.Lock()


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    global _WORKER_LOOP, _WORKER_THREAD
    with _LOOP_LOCK:
        if _WORKER_LOOP is None or not _WORKER_LOOP.is_running():
            _WORKER_LOOP = asyncio.new_event_loop()
            _WORKER_THREAD = threading.Thread(target=_WORKER_LOOP.run_forever, daemon=True)
            _WORKER_THREAD.start()
        return _WORKER_LOOP


class PrismaVectorStore:
    """Production vector store using Prisma ORM with the PostgreSQL pgvector extension."""

    def __init__(
        self,
        connection_string: str | None = None,
        dimension: int = 768,
    ) -> None:
        db_url = connection_string or os.getenv("DATABASE_URL")
        self.connection_string: str = (
            db_url if db_url else "postgresql://postgres:postgres@localhost:5432/zero_context"
        )
        self.dimension = dimension
        datasource: DatasourceOverride = {"url": self.connection_string}
        self._prisma = Prisma(datasource=datasource)
        self._connected = False

    async def aconnect(self) -> None:
        """Connect to the database via Prisma if not already connected."""
        if not self._connected:
            try:
                await self._prisma.connect()
                self._connected = True
            except Exception as err:
                raise StorageError(
                    f"Failed to connect to PostgreSQL via Prisma at '{self.connection_string}': {err}"
                ) from err

    async def adisconnect(self) -> None:
        """Disconnect Prisma client."""
        if self._connected:
            await self._prisma.disconnect()
            self._connected = False

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        """Execute an asynchronous Prisma coroutine synchronously on a persistent loop."""
        loop = _get_worker_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    async def aadd_batch(self, items: list[MemoryItem]) -> None:
        """Batch insert or update memory records using Prisma execute_raw."""
        if not items:
            return

        await self.aconnect()

        for item in items:
            vec_str = "[" + ",".join(str(x) for x in item.vector) + "]"
            meta_json = json.dumps(item.metadata)

            insert_sql = """
            INSERT INTO zero_context_memories (id, text, embedding, user_id, session_id, metadata, created_at, updated_at)
            VALUES ($1, $2, $3::vector, $4, $5, $6::jsonb, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                user_id = EXCLUDED.user_id,
                session_id = EXCLUDED.session_id,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """
            try:
                await self._prisma.execute_raw(
                    insert_sql,
                    item.id,
                    item.text,
                    vec_str,
                    item.user_id,
                    item.session_id,
                    meta_json,
                )
            except Exception as err:
                raise StorageError(f"Prisma failed to insert record '{item.id}': {err}") from err

    def add_batch(self, items: list[MemoryItem]) -> None:
        self._run_sync(self.aadd_batch(items))

    def add(self, item: MemoryItem) -> None:
        self.add_batch([item])

    async def asearch(
        self,
        query_vector: list[float],
        top_k: int = 5,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Perform nearest-neighbor vector search in pgvector using Prisma query_raw."""
        await self.aconnect()

        vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"

        try:
            if metadata_filter:
                search_sql = """
                SELECT id, text, user_id, session_id, metadata,
                       EXTRACT(EPOCH FROM created_at) AS created_at,
                       EXTRACT(EPOCH FROM updated_at) AS updated_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM zero_context_memories
                WHERE ($2::text IS NULL OR user_id = $2)
                  AND ($3::text IS NULL OR session_id = $3)
                  AND metadata @> $4::jsonb
                ORDER BY embedding <=> $1::vector ASC
                LIMIT $5
                """
                raw_rows: list[dict[str, Any]] = await self._prisma.query_raw(
                    search_sql,
                    vec_str,
                    user_id,
                    session_id,
                    json.dumps(metadata_filter),
                    top_k,
                )
            else:
                search_sql = """
                SELECT id, text, user_id, session_id, metadata,
                       EXTRACT(EPOCH FROM created_at) AS created_at,
                       EXTRACT(EPOCH FROM updated_at) AS updated_at,
                       1 - (embedding <=> $1::vector) AS score
                FROM zero_context_memories
                WHERE ($2::text IS NULL OR user_id = $2)
                  AND ($3::text IS NULL OR session_id = $3)
                ORDER BY embedding <=> $1::vector ASC
                LIMIT $4
                """
                raw_rows = await self._prisma.query_raw(
                    search_sql,
                    vec_str,
                    user_id,
                    session_id,
                    top_k,
                )

            results: list[SearchResult] = []
            for row in raw_rows:
                score = float(row.get("score", 0.0))
                if score < score_threshold:
                    continue

                meta = row.get("metadata", {})
                if isinstance(meta, str):
                    meta = json.loads(meta)

                item = MemoryItem(
                    id=row["id"],
                    text=row["text"],
                    vector=[],
                    user_id=row.get("user_id"),
                    session_id=row.get("session_id"),
                    metadata=meta if isinstance(meta, dict) else {},
                    created_at=float(row.get("created_at") or 0.0),
                    updated_at=float(row.get("updated_at") or 0.0),
                )
                results.append(SearchResult(memory=item, score=score))

            return results
        except Exception as err:
            raise StorageError(f"Prisma vector search failed: {err}") from err

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        return self._run_sync(
            self.asearch(
                query_vector=query_vector,
                top_k=top_k,
                user_id=user_id,
                session_id=session_id,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
            )
        )

    async def aget(self, memory_id: str) -> MemoryItem | None:
        await self.aconnect()
        try:
            record = await self._prisma.memory.find_unique(where={"id": memory_id})
            if not record:
                return None
            meta = record.metadata
            return MemoryItem(
                id=record.id,
                text=record.text,
                vector=[],
                user_id=record.userId,
                session_id=record.sessionId,
                metadata=meta if isinstance(meta, dict) else {},
                created_at=record.createdAt.timestamp(),
                updated_at=record.updatedAt.timestamp(),
            )
        except Exception as err:
            raise StorageError(f"Prisma failed to get memory '{memory_id}': {err}") from err

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._run_sync(self.aget(memory_id))

    async def aget_all(self, user_id: str | None = None) -> list[MemoryItem]:
        await self.aconnect()
        try:
            where_clause: MemoryWhereInput | None = {"userId": user_id} if user_id else None
            records = await self._prisma.memory.find_many(where=where_clause)
            return [
                MemoryItem(
                    id=rec.id,
                    text=rec.text,
                    vector=[],
                    user_id=rec.userId,
                    session_id=rec.sessionId,
                    metadata=rec.metadata if isinstance(rec.metadata, dict) else {},
                    created_at=rec.createdAt.timestamp(),
                    updated_at=rec.updatedAt.timestamp(),
                )
                for rec in records
            ]
        except Exception as err:
            raise StorageError(f"Prisma failed to fetch memories: {err}") from err

    def get_all(self, user_id: str | None = None) -> list[MemoryItem]:
        return self._run_sync(self.aget_all(user_id=user_id))

    async def adelete(self, memory_id: str) -> bool:
        await self.aconnect()
        try:
            res = await self._prisma.memory.delete(where={"id": memory_id})
            return res is not None
        except Exception:
            return False

    def delete(self, memory_id: str) -> bool:
        return self._run_sync(self.adelete(memory_id))

    async def aclear(self, user_id: str | None = None) -> int:
        await self.aconnect()
        try:
            where_del: MemoryWhereInput | None = {"userId": user_id} if user_id else None
            res = await self._prisma.memory.delete_many(where=where_del)
            return res
        except Exception as err:
            raise StorageError(f"Prisma failed to clear memories: {err}") from err

    def clear(self, user_id: str | None = None) -> int:
        return self._run_sync(self.aclear(user_id=user_id))
