
import streamlit as st

st.set_page_config(
    page_title="EDA Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        letter-spacing: -0.03em;
    }

    .stButton > button {
        font-family: 'DM Mono', monospace;
        border-radius: 2px;
        border: 1px solid #333;
        background: transparent;
        color: inherit;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #1a1a2e;
        border-color: #6c63ff;
        color: #6c63ff;
    }

    /* Sidebar nav items */
    [data-testid="stSidebarNav"] a {
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Status badge colours */
    .badge-running   { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade80; }
    .badge-queued    { background: #2a2a1a; color: #facc15; border: 1px solid #facc15; }
    .badge-completed { background: #1a2a3a; color: #60a5fa; border: 1px solid #60a5fa; }
    .badge-failed    { background: #3a1a1a; color: #f87171; border: 1px solid #f87171; }
    .badge-cancelled { background: #2a2a2a; color: #9ca3af; border: 1px solid #9ca3af; }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 2px;
        font-size: 0.75rem;
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* Metric delta override */
    [data-testid="stMetricDelta"] { font-size: 0.72rem; }

    /* Scrollable dataframe */
    .stDataFrame { border: 1px solid #2a2a2a; border-radius: 2px; }

    /* Divider */
    hr { border-color: #2a2a2a !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


_DEFAULTS: dict = {
    "job_id":           None,
    "session_id":       None,
    "job_status":       None,
    "hitl_pending":     False,
    "artifacts":        None,
    "budget_snapshot":  None,
    "orchestrator":     None,   
    "budget_controller": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.orchestrator is None:
    with st.spinner("Starting pipeline engine..."):
        try:
            from factory import bootstrap
            bootstrap()
        except Exception as exc:
            st.error(f"Failed to initialise pipeline engine: {exc}")
            st.exception(exc)
            st.stop()

            
with st.sidebar:
    st.markdown("## EDA Copilot")
    st.caption("Agentic data preparation pipeline")
    st.divider()

    job_id = st.session_state.get("job_id")
    if job_id:
        st.markdown(f"**Active job**")
        st.code(job_id[:16] + "…", language=None)

        status = st.session_state.get("job_status") or {}
        s = status.get("status", "—")
        css_class = {
            "RUNNING":   "badge-running",
            "QUEUED":    "badge-queued",
            "COMPLETED": "badge-completed",
            "FAILED":    "badge-failed",
            "CANCELLED": "badge-cancelled",
        }.get(s, "badge-queued")
        st.markdown(
            f'<span class="badge {css_class}">{s}</span>',
            unsafe_allow_html=True,
        )

        if st.session_state.hitl_pending:
            st.warning("Awaiting your review", icon="🛑")

    st.divider()
    st.caption("v0.1.0 · dataprep_agent")

st.title("EDA Copilot")
st.markdown(
    """
    An agentic ML data preparation pipeline with confidence routing,
    HITL gates, observability, and budget control.

    **Navigate using the sidebar** to:
    - **Upload** a dataset and submit a job
    - **Monitor** the state machine in real time
    - **Review** HITL-flagged plans
    - **Results** — download outputs and reports
    - **Budget** — token spend and cost breakdown
    """
)