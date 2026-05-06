"""
app/main.py — FastAPI application with Phase 1 LangGraph wiring.

Lifespan:
  1. Open DB connection pool
  2. Create DB schema (idempotent)
  3. Build LangGraph digest graph (with PostgresSaver)
  4. Build LangGraph alert graph
  5. Start APScheduler (daily digest @ 9 AM + alert check every N hours)

New endpoints vs Phase 0:
  GET /runs          — paginated run history from PostgreSQL
  GET /runs/{run_id} — single run detail (for time-travel debugging)
  GET /alert-now     — manually trigger the alert check
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import pytz

from app.config import settings
from app.graph.state import agent_state
from app.database import init_pool, close_pool, create_schema, fetch_run_history
from app.graph.digest_graph import build_graph, run_fintech_digest
from app.alert_graph import build_alert_graph, run_alert_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone=pytz.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. Database ────────────────────────────────────────────────────────
    await init_pool()
    await create_schema()

    # ── 2. LangGraph graphs ────────────────────────────────────────────────
    await build_graph()
    await build_alert_graph()

    # ── 3. Scheduler ────────────────────────────────────────────────────────
    tz = pytz.timezone(settings.USER_TIMEZONE)

    # Daily digest at 9:00 AM user's timezone
    scheduler.add_job(
        run_fintech_digest,
        CronTrigger(hour=9, minute=0, timezone=tz),
        id="daily_digest",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # Breaking alert check every ALERT_POLL_HOURS hours
    scheduler.add_job(
        run_alert_check,
        IntervalTrigger(hours=settings.ALERT_POLL_HOURS),
        id="alert_check",
        replace_existing=True,
        misfire_grace_time=300,
    )

    scheduler.start()
    logger.info(
        f"Scheduler started — daily digest at 9:00 AM {settings.USER_TIMEZONE}, "
        f"alert check every {settings.ALERT_POLL_HOURS}h"
    )

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    scheduler.shutdown()
    await close_pool()


app = FastAPI(
    title="FinTech Intelligence Agent",
    description="LangGraph-powered multi-agent fintech news briefing system",
    lifespan=lifespan,
)


# ── Dashboard ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    last_run = agent_state.get("last_run", "Never")
    last_status = agent_state.get("last_status", "—")
    stories_found = agent_state.get("stories_found", 0)
    history = agent_state.get("run_history", [])[-5:][::-1]

    rows = ""
    for r in history:
        icon = "✅" if "success" in r.get("status","") else "❌"
        dur = f"{r['duration_s']}s" if r.get("duration_s") else "—"
        rows += (
            f"<tr><td>{r['timestamp']}</td>"
            f"<td>{icon} {r['status'][:40]}</td>"
            f"<td>{r['stories']}</td>"
            f"<td>{dur}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html>
<head><title>FinTech Agent</title>
<style>
  body{{font-family:monospace;padding:2rem;background:#0a0a0a;color:#e0e0e0;max-width:900px;margin:0 auto}}
  h2{{color:#60a5fa}} h3{{color:#94a3b8;font-size:.9rem;margin-top:2rem}}
  a{{color:#60a5fa}} .ok{{color:#4ade80}} .err{{color:#f87171}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:.5rem}}
  td,th{{padding:6px 10px;border:1px solid #1e293b;text-align:left}}
  th{{background:#0f172a;color:#94a3b8}}
  .badge{{background:#1e3a5f;color:#60a5fa;padding:2px 8px;border-radius:12px;font-size:.75rem}}
</style>
</head>
<body>
<h2>🏦 FinTech Intelligence Agent</h2>
<span class="badge">LangGraph + PostgreSQL Checkpointing</span>

<p style="margin-top:1rem">
  Status: <strong class="{'ok' if 'success' in last_status else 'err'}">{last_status}</strong><br>
  Last run: {last_run}<br>
  Stories in last digest: {stories_found}<br>
  Schedule: 9:00 AM {settings.USER_TIMEZONE} · Alert check every {settings.ALERT_POLL_HOURS}h
</p>

<h3>ACTIONS</h3>
<p>
  <a href="/run-now">▶ Trigger digest now</a> &nbsp;|&nbsp;
  <a href="/alert-now">🚨 Trigger alert check now</a> &nbsp;|&nbsp;
  <a href="/preview">👁 Preview last email</a> &nbsp;|&nbsp;
  <a href="/runs">📋 Full run history (JSON)</a> &nbsp;|&nbsp;
  <a href="/health">❤ Health check</a>
</p>

<h3>RECENT RUNS</h3>
<table>
  <tr><th>Timestamp</th><th>Status</th><th>Stories</th><th>Duration</th></tr>
  {rows if rows else '<tr><td colspan="4">No runs yet</td></tr>'}
</table>
</body></html>"""


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from app.graph.digest_graph import _graph
    from app.database import get_pool
    return {
        "status": "ok",
        "scheduler_running": scheduler.running,
        "graph_ready": _graph is not None,
        "db_pool_open": get_pool().closed is False,
    }


# ── Manual triggers ──────────────────────────────────────────────────────────

@app.get("/run-now")
async def trigger_now(background_tasks: BackgroundTasks):
    """Manually trigger the daily digest pipeline."""
    background_tasks.add_task(run_fintech_digest)
    return {"message": "Digest triggered in background. Check /preview in ~60 seconds."}


@app.get("/alert-now")
async def trigger_alert(background_tasks: BackgroundTasks):
    """Manually trigger a breaking news alert check."""
    background_tasks.add_task(run_alert_check)
    return {"message": "Alert check triggered in background."}


# ── Preview ──────────────────────────────────────────────────────────────────

@app.get("/preview", response_class=HTMLResponse)
async def preview_last_email():
    """Preview the last generated email HTML in the browser."""
    html = agent_state.get("last_email_html")
    if not html:
        raise HTTPException(
            status_code=404,
            detail="No digest generated yet. Hit /run-now first.",
        )
    return html


# ── Run history ──────────────────────────────────────────────────────────────

@app.get("/runs")
async def list_runs(limit: int = 30):
    """Return paginated run history from PostgreSQL."""
    try:
        rows = await fetch_run_history(limit=limit)
        # Serialise datetimes for JSON
        for r in rows:
            for k, v in r.items():
                if isinstance(v, datetime):
                    r[k] = v.isoformat()
        return JSONResponse(content={"runs": rows, "count": len(rows)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))