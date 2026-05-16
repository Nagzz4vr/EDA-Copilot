from __future__ import annotations
 
from typing import Any, Dict, List
 
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh 

st.set_page_config(page_title="Job Monitor — EDA Copilot", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.03em; }

    .state-step {
        display: flex; flex-direction: column; align-items: center;
        font-size: 0.65rem; letter-spacing: 0.05em; text-transform: uppercase;
        padding: 6px 4px; gap: 4px;
    }
    .state-dot {
        width: 14px; height: 14px; border-radius: 50%;
        border: 2px solid currentColor;
    }
    .state-done  { color: #4ade80; }
    .state-done .state-dot  { background: #4ade80; }
    .state-active{ color: #facc15; }
    .state-active .state-dot { background: #facc15; animation: pulse 1s infinite; }
    .state-pending { color: #374151; }
    .state-pending .state-dot { background: transparent; }
    .state-failed { color: #f87171; }
    .state-failed .state-dot  { background: #f87171; }

    @keyframes pulse {
        0%,100% { opacity: 1; transform: scale(1); }
        50%      { opacity: 0.4; transform: scale(1.3); }
    }

    .hitl-banner {
        background: #3a1f00; border: 1px solid #f59e0b; border-radius: 4px;
        padding: 12px 18px; color: #fcd34d; margin-bottom: 12px;
        font-family: 'DM Mono', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

job_id = st.session_state.get("job_id")
if not job_id:
    st.warning("No active job — go to **Upload** first.")
    st.stop()

# Auto-refresh every 4 seconds for live updates
st_autorefresh(interval=4_000, key="monitor_refresh")

orchestrator = st.session_state.get("orchestrator")
status: Dict[str, Any] = {}

if orchestrator:
    from utils.api_client import get_status, is_hitl_pending
    try:
        status = get_status(orchestrator, job_id)
        st.session_state.job_status = status
    except Exception as e:
        st.error(f"Could not fetch status: {e}")
        st.stop()

    # Check HITL status
    try:
        st.session_state.hitl_pending = is_hitl_pending(job_id)
    except Exception:
        st.session_state.hitl_pending = False
else:
    status = st.session_state.get("job_status") or {}
    st.session_state.hitl_pending = st.session_state.get("hitl_pending", False)

st.title("Job Monitor")
st.caption(f"`{job_id}`")

# HITL Warning Banner
if st.session_state.get("hitl_pending"):
    st.markdown(
        '<div class="hitl-banner">'
        "⚠️  <strong>Pipeline paused</strong> — human approval required. "
        "Go to the <strong>HITL Review</strong> page."
        "</div>",
        unsafe_allow_html=True,
    )

STATES: List[str] = [
    "PERCEIVING", "SIGNALING", "MODELING", "PLANNING", "REVIEWING",
    "OPTIMIZING", "SIMULATING", "REFINING", "VALIDATING", "EXECUTING",
    "FINALIZING", "COMPLETED",
]

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}

# Get current state - prioritize 'state' over 'status'
current_state = status.get("state") or status.get("status") or "QUEUED"

STATE_PROGRESS = {
    "QUEUED": 0.0,
    "PERCEIVING": 0.08,
    "SIGNALING": 0.17,
    "MODELING": 0.25,
    "PLANNING": 0.33,
    "REVIEWING": 0.42,
    "OPTIMIZING": 0.50,
    "SIMULATING": 0.58,
    "REFINING": 0.67,
    "VALIDATING": 0.75,
    "EXECUTING": 0.83,
    "FINALIZING": 0.92,
    "COMPLETED": 1.0,
}

# Use state-based progress if explicit progress is 0 or not set
current_progress = status.get("progress")
if (current_progress is None or current_progress == 0.0) and current_state in STATE_PROGRESS:
    calculated_progress = STATE_PROGRESS.get(current_state, 0.0)
    status["progress"] = calculated_progress

st.subheader("Pipeline State")
cols = st.columns(len(STATES))

# Determine which states are done based on current position
try:
    current_idx = STATES.index(current_state) if current_state in STATES else -1
except ValueError:
    current_idx = -1

for i, (col, sname) in enumerate(zip(cols, STATES)):
    # Determine CSS class based on state
    if current_state == "FAILED":
        # If job failed, show all as pending except the failed state
        css = "state-failed" if sname == current_state else "state-pending"
    elif current_state == "CANCELLED":
        css = "state-pending"
    elif sname == current_state and current_state not in TERMINAL:
        # Currently active state
        css = "state-active"
    elif current_idx >= 0 and i < current_idx:
        # States that have been completed
        css = "state-done"
    elif current_state == "COMPLETED":
        # All states are done
        css = "state-done"
    else:
        # Pending states
        css = "state-pending"

    with col:
        st.markdown(
            f'<div class="state-step {css}">'
            f'<div class="state-dot"></div>'
            f"{sname.replace('_', ' ')}"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# Metrics Row
m1, m2, m3, m4, m5 = st.columns(5)

job_status_str = status.get("status", "—")
progress_pct   = round((status.get("progress") or 0.0) * 100, 1)

m1.metric("Status", job_status_str)
m2.metric("Progress", f"{progress_pct}%")
m3.metric("Actions Applied", status.get("actions_applied") or "—")
m4.metric("Rows Processed", status.get("rows_processed") or "—")

elapsed = "—"
budget_controller = st.session_state.get("budget_controller")
if budget_controller:
    from utils.api_client import get_budget_usage
    try:
        usage = get_budget_usage(budget_controller, job_id)
        if usage:
            elapsed = f"{usage.get('elapsed_seconds', '—')}s"
    except Exception:
        elapsed = "—"
m5.metric("Elapsed", elapsed)

st.divider()

# Confidence Chart
st.subheader("Confidence Over Steps")

from utils.api_client import get_trace_log

session_id = st.session_state.get("session_id", job_id)

try:
    trace_rows = get_trace_log(session_id)
except Exception as e:
    st.warning(f"Could not fetch trace log: {e}")
    trace_rows = []

if trace_rows:
    conf_data = [
        {
            "step": r.get("tool", ""),
            "confidence": r.get("confidence", 0.0),
            "timestamp": r.get("timestamp", "")
        }
        for r in trace_rows
        if r.get("confidence") is not None
    ]
    
    if conf_data:
        conf_df = pd.DataFrame(conf_data)
        st.line_chart(conf_df.set_index("step")["confidence"], use_container_width=True)
    else:
        st.caption("No confidence data yet.")
else:
    st.caption("Trace log empty — waiting for pipeline to start.")

st.divider()

# Trace Log Expander
with st.expander(" Trace Log", expanded=False):
    if trace_rows:
        # Select only available columns
        available_cols = ["timestamp", "tool", "intent", "confidence"]
        trace_df = pd.DataFrame(trace_rows)
        display_cols = [c for c in available_cols if c in trace_df.columns]
        
        if display_cols:
            st.dataframe(
                trace_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.json(trace_rows)
    else:
        st.caption("No trace entries yet.")

st.divider()

# Action Buttons
act_col1, act_col2, _ = st.columns([1, 1, 4])

# Cancel button - only show if job is running
if job_status_str in ("RUNNING", "QUEUED") and orchestrator:
    with act_col1:
        if st.button("🛑 Cancel Job", use_container_width=True):
            from utils.api_client import cancel_job
            try:
                cancel_job(orchestrator, job_id)
                st.success("Job cancelled")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to cancel: {e}")

# View Results button - only show if completed
if job_status_str == "COMPLETED":
    with act_col2:
        if st.button("View Results", use_container_width=True):
            st.switch_page("pages/4_Results.py")

# Error Display
if status.get("error"):
    st.error(f"**Job error:** {status['error']}")

# Debug info in expander (useful for development)
with st.expander(" Debug Info", expanded=False):
    st.json({
        "job_id": job_id,
        "status_object": status,
        "current_state": current_state,
        "hitl_pending": st.session_state.get("hitl_pending"),
        "session_state_keys": list(st.session_state.keys())
    })