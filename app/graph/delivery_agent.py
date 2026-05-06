"""
delivery_agent — Stage 5 of the digest graph.

Two responsibilities:
  1. Send the HTML email via Gmail API
  2. Persist the sent stories' embeddings to story_memory so future
     runs can deduplicate against them

Saving to memory only happens if the email send succeeds — we don't
want to record stories as "seen" if the user never received them.
"""

import logging
from app.graph.state import DigestState
from app.gmail import send_digest_email
from app.memory import save_stories_to_memory

logger = logging.getLogger(__name__)


async def delivery_agent(state: DigestState) -> dict:
    """Sends email, then saves sent stories to the memory store."""
    logger.info(f"[delivery_agent] run_id={state['run_id']}")

    if state.get("should_abort"):
        logger.info("[delivery_agent] Skipping — upstream abort")
        return {"email_sent": False}

    subject = state.get("email_subject", "FinTech Intelligence Digest")
    html = state.get("email_html", "")
    curated_stories = state.get("curated_stories", [])

    if not html:
        return {
            "email_sent": False,
            "should_abort": True,
            "abort_reason": "No HTML to send — builder_agent may have failed",
        }

    # --- Send email ---
    try:
        sent = send_digest_email(subject, html)
    except Exception as e:
        logger.error(f"[delivery_agent] Gmail send exception: {e}", exc_info=True)
        return {
            "email_sent": False,
            "errors": [f"delivery_agent send: {str(e)}"],
        }

    if not sent:
        return {
            "email_sent": False,
            "errors": ["delivery_agent: Gmail API returned failure"],
        }

    logger.info(f"[delivery_agent] Email sent: {subject}")

    # --- Save to memory (non-blocking failure OK) ---
    try:
        await save_stories_to_memory(curated_stories)
    except Exception as e:
        # Don't abort — email already sent, memory is best-effort
        logger.warning(f"[delivery_agent] Memory save failed (non-critical): {e}")
        return {
            "email_sent": True,
            "errors": [f"delivery_agent memory save: {str(e)}"],
        }

    return {"email_sent": True}