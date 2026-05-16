from __future__ import annotations
 
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
 
from config import HITL_DB_PATH, LEDGER_DIR
from core.hitl.hitl_store import HitlStore
import logging
import traceback

from observability.trace_logger import TraceLogger
logger = logging.getLogger(__name__)

_hitl_store: Optional[HitlStore] = None
 
 
def get_hitl_store() -> HitlStore:
    global _hitl_store
    if _hitl_store is None:
        _hitl_store = HitlStore(db_path=HITL_DB_PATH)
    return _hitl_store

def _run_async(coro):

    import streamlit as _st

    loop = _st.session_state.get("bg_loop")
    if loop is None or not loop.is_running():
        raise RuntimeError(
            "Background event loop not found in st.session_state.bg_loop. "
            "Make sure factory.bootstrap() completed successfully."
        )
    future = __import__("asyncio").run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)

def submit_job(
    orchestrator,
    base_dir:str,
    file_path: str,
    target_col: str,
    budget_overrides: Optional[Dict[str, Any]] = None,
    max_iterations: int = 3,
) -> str:
    from control.orchestrator import SubmitRequest

    request = SubmitRequest(
    file_path=file_path,
    base_dir=base_dir,
    target_column=target_col,
    max_iterations=max_iterations,
    budget_tokens=budget_overrides.get("max_tokens_total", 25_000)
    if budget_overrides
    else 25_000,
)
    status = _run_async(orchestrator.submit_job(request))
    return status.job_id

def get_status(orchestrator, job_id: str) -> Dict[str, Any]:
    status = _run_async(orchestrator.get_status(job_id))
    return status.model_dump()

def cancel_job(orchestrator, job_id: str) -> Dict[str, Any]:
    status = _run_async(orchestrator.cancel_job(job_id))
    return status.model_dump()

def list_jobs(orchestrator) -> List[Dict[str, Any]]:
    jobs = _run_async(orchestrator.list_jobs())
    return [j.model_dump() for j in jobs]


def get_hitl_payload(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns the pending HITL row for this job, or None.
    Deserialises plan_json / risk_json blobs automatically.
    """
    store = get_hitl_store()
    row = store.read_request(job_id)
    if row is None or row.get("status") != "PENDING":
        return None
    return row

def submit_hitl_decision(
    job_id: str,
    action: str,                  # "APPROVED" | "REJECTED"
    reason: Optional[str] = None,
) -> None:
    store = get_hitl_store()
    store.write_decision(job_id=job_id, action=action, reason=reason)

def is_hitl_pending(job_id: str) -> bool:
    store = get_hitl_store()
    return store.is_pending(job_id)

def get_budget_usage(budget_controller, job_id: str) -> Optional[Dict[str, Any]]:
    try:
        return budget_controller.get_usage_summary(job_id)
    except KeyError:
        return None

def get_budget_remaining(budget_controller, job_id: str) -> Optional[Dict[str, Any]]:
    try:
        return budget_controller.remaining_budget(job_id)
    except KeyError:
        return None
    
def get_ledger_rows(session_id: str) -> List[Dict[str, Any]]:
    """Read all JSONL rows for a session from the ledger directory."""
    ledger_path = Path(LEDGER_DIR) / f"{session_id}.jsonl"
    if not ledger_path.exists():
        return []
    rows = []
    with open(ledger_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows

def get_artifacts(job_id: str) -> Optional[Dict[str, Any]]:
    base = Path("outputs")

    dataset = base / "datasets" / f"{job_id}.parquet"
    pipeline = base / "pipelines" / f"{job_id}_pipeline.py"
    report = base / "reports" / f"{job_id}_report.json"

    return {
        "dataset": {"path": str(dataset)},
        "pipeline": {"path": str(pipeline)},
        "report": str(report),
    }
    
def get_trace_log(session_id: str) -> List[Dict[str, Any]]:
    """
    Reads the trace logger JSONL for a session.
    Adjust path to match your TraceLogger's log_dir.
    """
    trace_path = Path("traces") / f"{session_id}.jsonl"
    print(trace_path)
    if not trace_path.exists():
        return []
    rows = []
    with open(trace_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows