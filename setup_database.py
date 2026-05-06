#!/usr/bin/env python3
"""
Run this ONCE before deploying to Render (or let the app auto-run it on startup).
The app's lifespan calls create_schema() automatically, so this script is mainly
useful for verifying your DATABASE_URL is correct before the first deploy.

Usage:
    DATABASE_URL=postgresql://... python scripts/setup_database.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def main():
    from app.database import init_pool, create_schema, close_pool

    print("Connecting to database...")
    await init_pool()

    print("Creating schema...")
    await create_schema()

    print("\n✅ Database schema ready. Tables created:")
    print("  - story_memory     (pgvector embeddings for deduplication)")
    print("  - run_history      (persistent agent run log)")
    print("  - alert_history    (sent breaking alerts — prevents re-sending)")
    print("\nLangGraph will create its own checkpoint tables on first graph run.")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())