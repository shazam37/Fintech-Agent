"""
curator_agent — Stage 3 of the digest graph.

The LLM node. Takes novel_stories from the memory agent,
calls Groq to rank, summarise, and generate a subject line.

This is the most expensive node (API call). If Groq fails,
falls back to raw snippets so the digest still delivers value.
"""

import logging
from app.state import DigestState
from app.llm import process_stories
from app.config import settings

logger = logging.getLogger(__name__)


def curator_agent(state: DigestState) -> dict:
    """
    Sends novel stories to Groq for curation. Handles JSON parse
    errors and API errors via the existing _fallback_format in llm.py.
    """
    logger.info(f"[curator_agent] run_id={state['run_id']}")

    if state.get("should_abort"):
        logger.info("[curator_agent] Skipping — upstream abort")
        return {}

    novel_stories = state.get("novel_stories", [])
    if not novel_stories:
        return {
            "should_abort": True,
            "abort_reason": "No novel stories to curate",
        }

    try:
        digest = process_stories(novel_stories)
        curated = digest.get("stories", [])
        subject = digest.get("subject", "FinTech Intelligence Digest")

        logger.info(f"[curator_agent] LLM selected {len(curated)} stories")

        if len(curated) < settings.MIN_STORIES_BEFORE_SEND:
            return {
                "curated_stories": curated,
                "email_subject": subject,
                "should_abort": True,
                "abort_reason": (
                    f"Only {len(curated)} stories curated "
                    f"(minimum is {settings.MIN_STORIES_BEFORE_SEND})"
                ),
            }

        return {
            "curated_stories": curated,
            "email_subject": subject,
        }

    except Exception as e:
        logger.error(f"[curator_agent] Fatal: {e}", exc_info=True)
        return {
            "should_abort": True,
            "abort_reason": f"curator_agent exception: {str(e)[:120]}",
            "errors": [f"curator_agent: {str(e)}"],
        }