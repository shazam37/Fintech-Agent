"""
digest_graph.py — The main LangGraph StateGraph for daily digest runs.

Graph topology:

  START
    │
    ▼
  news_agent          ← Tavily search (parallel queries)
    │
    ▼
  memory_agent        ← pgvector deduplication (async)
    │
    ▼
  curator_agent       ← Groq LLM: rank, summarise, subject line
    │
    ▼
  builder_agent       ← HTML rendering
    │
    ▼
  delivery_agent      ← Gmail send + save to memory store (async)
    │
    ▼
  calendar_agent      ← Calendar event + DB run log (async)
    │
    ▼
  END

Each node checks state["should_abort"] and skips its work if True,
passing control to the next node (which eventually reaches calendar_agent
for cleanup/logging). This avoids complex conditional edges while still
guaranteeing the audit trail is always written.

The PostgresSaver checkpointer snapshots state after every node.
If the process crashes mid-run, the next trigger resumes from the
last completed node rather than starting over.
"""

import logging
import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.state import DigestState
from app.graph.news_agent import news_agent
from app.graph.memory_agent import memory_agent
from app.graph.curator_agent import curator_agent
from app.graph.builder_agent import builder_agent
from app.graph.delivery_agent import delivery_agent
from app.graph.calendar_agent import calendar_agent
from app.config import settings

logger = logging.getLogger(__name__)

# Module-level compiled graph — built once in build_graph(), reused for all runs
_graph = None
_checkpointer = None


async def build_graph():
    """
    Compile the LangGraph StateGraph with the PostgresSaver checkpointer.
    Call once at app startup (inside lifespan).
    """
    global _graph, _checkpointer

    # The AsyncPostgresSaver manages its own connection pool internally.
    # It creates the langgraph_checkpoints table on first use.
    _checkpointer = AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL)
    await _checkpointer.setup()  # Creates LangGraph's internal checkpoint tables

    graph_builder = StateGraph(DigestState)

    # Register all nodes
    graph_builder.add_node("news_agent", news_agent)
    graph_builder.add_node("memory_agent", memory_agent)
    graph_builder.add_node("curator_agent", curator_agent)
    graph_builder.add_node("builder_agent", builder_agent)
    graph_builder.add_node("delivery_agent", delivery_agent)
    graph_builder.add_node("calendar_agent", calendar_agent)

    # Linear edges — all nodes run sequentially
    # Abort logic lives inside each node (checks should_abort), not in edges,
    # so the graph always reaches calendar_agent for cleanup
    graph_builder.add_edge(START, "news_agent")
    graph_builder.add_edge("news_agent", "memory_agent")
    graph_builder.add_edge("memory_agent", "curator_agent")
    graph_builder.add_edge("curator_agent", "builder_agent")
    graph_builder.add_edge("builder_agent", "delivery_agent")
    graph_builder.add_edge("delivery_agent", "calendar_agent")
    graph_builder.add_edge("calendar_agent", END)

    _graph = graph_builder.compile(checkpointer=_checkpointer)
    logger.info("LangGraph digest graph compiled with PostgresSaver checkpointer")
    return _graph


def get_graph():
    if _graph is None:
        raise RuntimeError("Graph not built. Call build_graph() at startup.")
    return _graph


async def run_fintech_digest():
    """
    Entry point for the daily digest scheduler job.
    Creates a fresh run_id (= LangGraph thread_id) and invokes the graph.

    Each run is fully isolated — a new thread_id means a new checkpoint chain.
    Old checkpoints are kept for time-travel debugging but don't interfere.
    """
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    logger.info(f"=== Starting digest run {run_id} ===")

    initial_state: DigestState = {
        "run_id": run_id,
        "run_type": "daily_digest",
        "started_at": started_at,
        "raw_stories": [],
        "novel_stories": [],
        "curated_stories": [],
        "email_subject": "",
        "email_html": "",
        "email_sent": False,
        "calendar_logged": False,
        "errors": [],
        "should_abort": False,
        "abort_reason": "",
    }

    config = {"configurable": {"thread_id": run_id}}

    try:
        graph = get_graph()
        final_state = await graph.ainvoke(initial_state, config=config)

        email_sent = final_state.get("email_sent", False)
        errors = final_state.get("errors", [])
        stories = len(final_state.get("curated_stories", []))

        logger.info(
            f"=== Digest run {run_id} complete: "
            f"sent={email_sent}, stories={stories}, errors={len(errors)} ==="
        )

        if errors:
            logger.warning(f"Run {run_id} non-fatal errors: {errors}")

    except Exception as e:
        logger.exception(f"=== Digest run {run_id} CRASHED: {e} ===")
        # Attempt to persist the failure to DB even when the graph itself crashes
        try:
            from app.database import upsert_run
            await upsert_run(
                run_id=run_id,
                status=f"crashed: {str(e)[:120]}",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
        except Exception:
            pass  # Best effort — don't mask the original exception