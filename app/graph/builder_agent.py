"""
builder_agent — Stage 4 of the digest graph.

Pure function: takes curated stories + subject line, renders HTML.
No I/O, no API calls — just template rendering. Should never fail.
"""

import logging
from app.state import DigestState
from app.email_builder import build_email_html

logger = logging.getLogger(__name__)


def builder_agent(state: DigestState) -> dict:
    """Renders the HTML email and caches it for the /preview endpoint."""
    logger.info(f"[builder_agent] run_id={state['run_id']}")

    if state.get("should_abort"):
        logger.info("[builder_agent] Skipping — upstream abort")
        return {}

    try:
        digest = {
            "subject": state.get("email_subject", "FinTech Intelligence Digest"),
            "stories": state.get("curated_stories", []),
        }
        html = build_email_html(digest)

        # Cache for /preview endpoint on the dashboard
        agent_state["last_email_html"] = html

        logger.info(f"[builder_agent] HTML built ({len(html)} chars)")
        return {"email_html": html}

    except Exception as e:
        logger.error(f"[builder_agent] Fatal: {e}", exc_info=True)
        return {
            "should_abort": True,
            "abort_reason": f"builder_agent exception: {str(e)[:120]}",
            "errors": [f"builder_agent: {str(e)}"],
        }