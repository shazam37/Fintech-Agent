"""
Story memory — semantic deduplication using pgvector.

How it works:
  1. Before each digest run, the memory agent fetches embeddings of all stories
     seen in the last MEMORY_LOOKBACK_DAYS days from PostgreSQL.
  2. Candidate stories have their titles embedded locally (all-MiniLM-L6-v2, 384-dim).
  3. Any candidate whose embedding is within SIMILARITY_THRESHOLD cosine distance
     of a stored story is dropped as a near-duplicate.
  4. After the digest is sent, the selected stories' embeddings are saved.

The model (all-MiniLM-L6-v2) is 80MB and runs fully offline — no API calls,
no cost, works on Render's free tier.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from app.config import settings
from app.database import get_conn

logger = logging.getLogger(__name__)

# Lazy-loaded — only imported when first needed so startup is fast
_model = None


def _get_model():
    """Load the sentence-transformer model (once, then cached)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts. Returns a (N, 384) float32 array.
    Uses title text only — titles are the stable identifier for a story.
    """
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalised vectors."""
    return float(np.dot(a, b))


async def filter_seen_stories(candidate_stories: list[dict]) -> list[dict]:
    """
    Given a list of candidate story dicts, return only those NOT already seen
    in the last MEMORY_LOOKBACK_DAYS days.

    Strategy:
      - First, hard-filter by URL (exact match)
      - Then, soft-filter by title embedding similarity
    """
    if not candidate_stories:
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.MEMORY_LOOKBACK_DAYS)

    # Fetch recent embeddings from DB
    async with get_conn() as conn:
        rows = await (await conn.execute(
            """
            SELECT url, embedding::text
            FROM story_memory
            WHERE created_at >= %s
            """,
            (cutoff,),
        )).fetchall()

    seen_urls = {r[0] for r in rows}

    # Parse stored embeddings back to numpy
    stored_embeddings: list[np.ndarray] = []
    for _, emb_text in rows:
        if emb_text:
            vec = np.array(
                [float(x) for x in emb_text.strip("[]").split(",")],
                dtype=np.float32,
            )
            stored_embeddings.append(vec)

    # Hard-filter by URL
    url_filtered = [s for s in candidate_stories if s["url"] not in seen_urls]

    if not url_filtered:
        logger.info("All candidates already seen (URL match) — nothing new")
        return []

    if not stored_embeddings:
        # First run — nothing to compare against
        return url_filtered

    # Soft-filter by embedding similarity
    titles = [s["title"] for s in url_filtered]
    candidate_embeddings = embed(titles)
    stored_matrix = np.stack(stored_embeddings)  # (M, 384)

    novel_stories = []
    for i, story in enumerate(url_filtered):
        cand_vec = candidate_embeddings[i]  # (384,)
        # Cosine similarities against all stored embeddings (dot product since both are normalised)
        sims = stored_matrix @ cand_vec      # (M,)
        max_sim = float(sims.max())

        if max_sim >= settings.SIMILARITY_THRESHOLD:
            logger.debug(
                f"Deduped (similarity={max_sim:.3f}): {story['title'][:60]}"
            )
        else:
            novel_stories.append(story)

    logger.info(
        f"Memory filter: {len(candidate_stories)} candidates → "
        f"{len(novel_stories)} novel stories "
        f"(dropped {len(candidate_stories) - len(novel_stories)})"
    )
    return novel_stories


async def save_stories_to_memory(stories: list[dict]):
    """
    Persist the digested stories' embeddings to PostgreSQL.
    Called after a successful digest send.
    """
    if not stories:
        return

    titles = [s["title"] for s in stories]
    embeddings = embed(titles)

    async with get_conn() as conn:
        for story, emb in zip(stories, embeddings):
            emb_list = emb.tolist()
            await conn.execute(
                """
                INSERT INTO story_memory (url, title, source, embedding)
                VALUES (%s, %s, %s, %s::vector)
                ON CONFLICT (url) DO NOTHING
                """,
                (story["url"], story["title"], story.get("source", ""), str(emb_list)),
            )
        await conn.commit()

    logger.info(f"Saved {len(stories)} story embeddings to memory")