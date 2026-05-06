"""
Database module.

Manages:
  - psycopg connection pool (shared across the app)
  - Schema creation on startup (idempotent)
  - run_history table (persistent across restarts, unlike in-memory state)
  - story_memory table + pgvector extension (semantic deduplication)

The LangGraph PostgresSaver uses its own internal tables (langgraph_checkpoints)
managed automatically by the library. We don't touch those here.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import psycopg
from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level pool — initialised in lifespan, used everywhere
_pool: AsyncConnectionPool | None = None


async def init_pool() -> AsyncConnectionPool:
    """Create the async connection pool. Called once at app startup."""
    global _pool
    _pool = AsyncConnectionPool(
        conninfo=settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        open=False,
    )
    await _pool.open()
    logger.info("Database connection pool opened")
    return _pool


async def close_pool():
    """Close the pool. Called at app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("Database connection pool closed")


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised. Call init_pool() first.")
    return _pool


@asynccontextmanager
async def get_conn() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Async context manager: yields a connection from the pool."""
    async with get_pool().connection() as conn:
        yield conn


async def create_schema():
    """
    Idempotent schema creation. Safe to run every startup.
    Creates:
      - pgvector extension
      - story_memory table (stores embeddings for deduplication)
      - run_history table (persistent agent run log)
      - alert_history table (tracks sent breaking alerts to avoid repeats)
    """
    async with get_conn() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS story_memory (
                id          SERIAL PRIMARY KEY,
                url         TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                source      TEXT,
                embedding   vector(384),          -- all-MiniLM-L6-v2 dimension
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Index for fast cosine similarity search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS story_memory_embedding_idx
            ON story_memory
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 50)
        """)

        # Index for fast time-based lookups (memory window queries)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS story_memory_created_idx
            ON story_memory (created_at DESC)
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS run_history (
                id          SERIAL PRIMARY KEY,
                run_id      TEXT UNIQUE NOT NULL,
                status      TEXT NOT NULL,
                stories     INTEGER DEFAULT 0,
                subject     TEXT,
                error_msg   TEXT,
                started_at  TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                duration_s  FLOAT
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id          SERIAL PRIMARY KEY,
                url         TEXT UNIQUE NOT NULL,
                title       TEXT NOT NULL,
                urgency     INTEGER NOT NULL,
                sent_at     TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        await conn.commit()
        logger.info("Database schema ready")


async def upsert_run(
    run_id: str,
    status: str,
    stories: int = 0,
    subject: str | None = None,
    error_msg: str | None = None,
    started_at=None,
    finished_at=None,
    duration_s: float | None = None,
):
    """Insert or update a run_history row."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO run_history
                (run_id, status, stories, subject, error_msg, started_at, finished_at, duration_s)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                status      = EXCLUDED.status,
                stories     = EXCLUDED.stories,
                subject     = EXCLUDED.subject,
                error_msg   = EXCLUDED.error_msg,
                finished_at = EXCLUDED.finished_at,
                duration_s  = EXCLUDED.duration_s
            """,
            (run_id, status, stories, subject, error_msg, started_at, finished_at, duration_s),
        )
        await conn.commit()


async def fetch_run_history(limit: int = 30) -> list[dict]:
    """Return the most recent runs for the dashboard."""
    async with get_conn() as conn:
        rows = await conn.execute(
            "SELECT * FROM run_history ORDER BY started_at DESC LIMIT %s",
            (limit,),
        )
        cols = [d.name for d in rows.description]
        return [dict(zip(cols, row)) for row in await rows.fetchall()]


async def was_alert_sent(url: str) -> bool:
    """Check if a breaking alert for this URL was already sent."""
    async with get_conn() as conn:
        row = await (await conn.execute(
            "SELECT 1 FROM alert_history WHERE url = %s", (url,)
        )).fetchone()
        return row is not None


async def record_alert_sent(url: str, title: str, urgency: int):
    """Mark a breaking alert as sent so we don't re-send it."""
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO alert_history (url, title, urgency)
            VALUES (%s, %s, %s)
            ON CONFLICT (url) DO NOTHING
            """,
            (url, title, urgency),
        )
        await conn.commit()

async def fetch_run_by_id(run_id: str) -> dict | None:
    """Fetch a single run by run_id (for preview + debugging)."""
    async with get_conn() as conn:
        result = await conn.execute(
            "SELECT * FROM run_history WHERE run_id = %s",
            (run_id,),
        )

        row = await result.fetchone()

        if not row:
            return None

        cols = [d.name for d in result.description]
        return dict(zip(cols, row))