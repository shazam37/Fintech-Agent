"""
news_agent — Stage 1 of the digest graph.

Fetches raw stories from Tavily using the existing search module.
This node is purely I/O bound; all LLM logic lives in curator_agent.

On failure: sets should_abort = True with a reason so downstream
nodes can skip gracefully rather than crash.
"""

import logging
from app.graph.state import DigestState
from app.search import fetch_fintech_news, SEARCH_QUERIES
from app.config import settings
from tavily import TavilyClient

logger = logging.getLogger(__name__)


def news_agent(state: DigestState) -> dict:
    """
    Fetches fintech news. Also supports watchlist queries
    (passed via state in future phases — currently uses default queries).
    """
    logger.info(f"[news_agent] run_id={state['run_id']}")

    try:
        raw_stories = fetch_fintech_news()
        logger.info(f"[news_agent] Fetched {len(raw_stories)} raw stories")

        if not raw_stories:
            return {
                "raw_stories": [],
                "should_abort": True,
                "abort_reason": "No stories returned from search — Tavily may be rate-limited or all results were excluded",
            }

        return {"raw_stories": raw_stories}

    except Exception as e:
        logger.error(f"[news_agent] Fatal: {e}", exc_info=True)
        return {
            "raw_stories": [],
            "should_abort": True,
            "abort_reason": f"news_agent exception: {str(e)[:120]}",
            "errors": [f"news_agent: {str(e)}"],
        }