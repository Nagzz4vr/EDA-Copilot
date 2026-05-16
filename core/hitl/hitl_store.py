from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from app.config import HITL_DB_PATH

_LOCK = Lock()

_DDL = """
CREATE TABLE IF NOT EXISTS hitl_requests (
    job_id          TEXT PRIMARY KEY,
    state_uuid      TEXT,
    plan_json       TEXT    NOT NULL,
    risk_json       TEXT    NOT NULL,
    diff_json       TEXT,
    status          TEXT    NOT NULL DEFAULT 'PENDING',
    decision_reason TEXT,
    created_at      REAL    NOT NULL,
    resolved_at     REAL
);
"""

class HitlStore:
    def __init__(self, db_path: str = HITL_DB_PATH) -> None:

        try:

            path_obj = Path(db_path).expanduser().resolve()


            path_obj.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._path = str(path_obj)

            self._init_db()

        except Exception as e:

            print(
                f"[HitlStore] Failed to initialize DB at "
                f"{db_path!r}: {e}"
            )

            fallback_path = Path("fallback_hitl_store.db").resolve()

            print(
                f"[HitlStore] Falling back to "
                f"{fallback_path}"
            )

            self._path = str(fallback_path)

            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _LOCK:
            with self._connect() as conn:
                conn.execute(_DDL)
                conn.commit()

    def write_request(
        self,
        *,
        job_id: str,
        state_uuid: str,
        plan_dict: Dict[str, Any],
        risk_result: Dict[str, Any],
        diff_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Called by HITLGate before entering the poll loop.
        Inserts or replaces a PENDING row for this job.
        """
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO hitl_requests
                        (job_id, state_uuid, plan_json, risk_json, diff_json,
                        status, decision_reason, created_at, resolved_at)
                    VALUES (?, ?, ?, ?, ?, 'PENDING', NULL, ?, NULL)
                    """,
                    (
                        job_id,
                        state_uuid,
                        json.dumps(plan_dict,   default=str),
                        json.dumps(risk_result, default=str),
                        json.dumps(diff_data,   default=str) if diff_data else None,
                        time.time(),
                    ),
                )
                conn.commit()

 
    def write_decision(
        self,
        job_id: str,
        action: str,           # "APPROVED" | "REJECTED"
        reason: Optional[str] = None,
    ) -> None:
        """
        Called by the Streamlit page when the human clicks Approve / Reject.
        MetaAgent's poll loop will pick this up on the next tick.
        """
        if action not in ("APPROVED", "REJECTED"):
            raise ValueError(f"action must be APPROVED or REJECTED, got {action!r}")
 
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE hitl_requests
                    SET status=?, decision_reason=?, resolved_at=?
                    WHERE job_id=?
                    """,
                    (action, reason, time.time(), job_id),
                )
                conn.commit()

    def read_request(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the full row as a plain dict, or None if not found."""
        with _LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM hitl_requests WHERE job_id = ?", (job_id,)
                ).fetchone()
        if row is None:
            return None
        d = dict(row)

        for key in ("plan_json", "risk_json", "diff_json"):
            raw = d.get(key)
            if raw:
                try:
                    d[key] = json.loads(raw)
                except json.JSONDecodeError:
                    pass 
        return d

    def is_pending(self, job_id: str) -> bool:
        row = self.read_request(job_id)
        return row is not None and row.get("status") == "PENDING"

    def is_resolved(self, job_id: str) -> bool:
        row = self.read_request(job_id)
        return row is not None and row.get("status") != "PENDING"

    def clear(self, job_id: str) -> None:
        """Remove the row once MetaAgent has consumed the decision."""
        with _LOCK:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM hitl_requests WHERE job_id = ?", (job_id,)
                )
                conn.commit()

    def all_pending(self) -> list[Dict[str, Any]]:
        """Return all PENDING rows — useful for a job-list view."""
        with _LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM hitl_requests WHERE status = 'PENDING'"
                ).fetchall()
        return [dict(r) for r in rows]