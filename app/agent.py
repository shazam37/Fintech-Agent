"""
app/agent.py

Backwards-compatibility shim.
The actual orchestration now lives in app/graph/digest_graph.py.
This file re-exports run_fintech_digest so any existing import
(e.g. from app.agent import run_fintech_digest) continues to work.
"""

from app.graph.digest_graph import run_fintech_digest  # noqa: F401

__all__ = ["run_fintech_digest"]