"""Audit schema and append-only audit log for the HITL churn-risk workflow.

The audit trail answers six questions about every decision the system makes:
which agent decided, what it proposed, how confident it was, who reviewed it,
what the human decided, and when it happened.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "audit_log.json"

AGENT_ID = "churn-risk-agent"

# reviewer_id value used when no human was involved, so "was this reviewed by a
# person?" stays a single-field question instead of a join across fields.
SYSTEM_REVIEWER = "system:auto"


class AuditEntry(BaseModel):
    """One traceable decision: what the agent proposed, what a human did about it."""

    # --- the six fields required by the lab spec ---
    timestamp: str
    agent_id: str
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    reviewer_id: str
    decision: str

    # --- extra context, so a log line is readable months later ---
    customer_id: str = ""
    reasoning: str = ""
    route_reason: str = ""
    executed: bool = False
    details: str = ""


def now_iso() -> str:
    """Timestamp in the format the lab shows: 2026-08-29T09:00:00."""
    return datetime.now().isoformat(timespec="seconds")


def load_audit_log() -> list[dict]:
    """Return the full audit history, or an empty list if there is none yet."""
    if not AUDIT_LOG_PATH.exists():
        return []
    try:
        raw = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A corrupt log is a real problem, but it must not take the workflow
        # down: fall back to empty rather than crashing the graph mid-run.
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def append_audit_entry(entry: AuditEntry) -> AuditEntry:
    """Read history, append one entry, write the whole list back.

    Read-append-write is what keeps earlier entries intact; writing the new
    object straight over the file is the classic way to lose the trail.
    """
    entries = load_audit_log()
    entries.append(entry.model_dump())
    AUDIT_LOG_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return entry
