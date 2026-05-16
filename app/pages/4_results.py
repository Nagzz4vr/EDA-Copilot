
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Results — EDA Copilot", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.03em; }
    .stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace; font-size:0.8rem;
                                    letter-spacing:0.06em; text-transform:uppercase; }
    .violation-item { background:#1f1f2e; border-left:3px solid #f59e0b;
                      padding:8px 12px; margin-bottom:6px; border-radius:0 3px 3px 0;
                      font-size:0.8rem; color:#fcd34d; }
    .action-log-row { background:#111827; border:1px solid #1f2937;
                      padding:6px 10px; border-radius:3px; margin-bottom:4px;
                      font-size:0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Guard ─────────────────────────────────────────────────────────────────
job_id = st.session_state.get("job_id")
if not job_id:
    st.warning("No active job — go to **Upload** first.")
    st.stop()

orchestrator = st.session_state.get("orchestrator")
if orchestrator:
    from utils.api_client import get_status
    try:
        status = get_status(orchestrator, job_id)
        st.session_state.job_status = status
    except Exception:
        status = st.session_state.get("job_status") or {}
else:
    status = st.session_state.get("job_status") or {}

current_status = status.get("status", "UNKNOWN")

st.title("Results")
st.caption(f"Job `{job_id}`")

if current_status != "COMPLETED":
    st.info(f"Job status is **{current_status}** — results will appear here once the job completes.")
    if st.button("↩  Back to Monitor"):
        st.switch_page("pages/2_Job_Monitor.py")
    st.stop()

# ── Load artifacts ────────────────────────────────────────────────────────
from utils.api_client import get_artifacts

@st.cache_data(show_spinner="Loading artifacts…", ttl=60)
def _load_artifacts(jid: str) -> Optional[Dict[str, Any]]:
    return get_artifacts(jid)


artifacts = _load_artifacts(job_id)

if artifacts is None:
    st.warning(
        "Artifacts manifest not found. "
        "Check that `report_generator` wrote to `outputs/{job_id}/manifest.json`."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_data, tab_pipeline, tab_report = st.tabs(["📊  Dataset", "🔩  Pipeline", "📋  Report"])

# ── Dataset tab ───────────────────────────────────────────────────────────
with tab_data:
    dataset_meta = artifacts.get("dataset") or {}
    dataset_path = dataset_meta.get("path")

    if dataset_path and Path(dataset_path).exists():
        @st.cache_data(show_spinner="Loading dataset…")
        def _load_df(path: str) -> pd.DataFrame:
            ext = Path(path).suffix.lower()
            if ext == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)

        df_out = _load_df(dataset_path)

        r1, r2, r3 = st.columns(3)
        r1.metric("Rows",    f"{len(df_out):,}")
        r2.metric("Columns", len(df_out.columns))
        r3.metric("Size",    f"{df_out.memory_usage(deep=True).sum() / 1024:.1f} KB")

        st.subheader("Preview (first 50 rows)")
        st.dataframe(df_out.head(50), use_container_width=True)

        # Download
        csv_bytes = df_out.to_csv(index=False).encode()
        st.download_button(
            "⬇  Download CSV",
            data=csv_bytes,
            file_name=f"{job_id}_processed.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning(f"Dataset file not found at `{dataset_path}`.")

# ── Pipeline tab ──────────────────────────────────────────────────────────
with tab_pipeline:
    pipeline_meta = artifacts.get("pipeline") or {}
    pipeline_path = pipeline_meta.get("path")

    st.subheader("Pipeline (.py)")

    if pipeline_path and Path(pipeline_path).exists():
        with open(pipeline_path, "r", encoding="utf-8") as f:
            pipeline_code = f.read()

        st.code(pipeline_code, language="python")

        st.download_button(
            "Download pipeline.py",
            data=pipeline_code.encode("utf-8"),
            file_name=f"{job_id}_pipeline.py",
            mime="text/x-python",
            use_container_width=True,
        )
    else:
        st.caption(f"Pipeline file not found at `{pipeline_path}`")

# ── Report tab ────────────────────────────────────────────────────────────
with tab_report:
    report_path_val = artifacts.get("report")

    report: Dict[str, Any] = {}
    if report_path_val and Path(str(report_path_val)).exists():
        with open(str(report_path_val), "r", encoding="utf-8") as fh:
            report = json.load(fh)
    elif isinstance(report_path_val, dict):
        report = report_path_val

    if not report:
        st.caption("Report file not found or empty.")
    else:
        exec_metrics   = report.get("execution_metrics", {})
        conf_summary   = report.get("confidence_summary", {})
        signal_summary = report.get("signal_summary", {})
        val_summary    = report.get("validation_summary", {})

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actions Applied",  len(exec_metrics.get("actions_applied", [])))
        c2.metric("Final Confidence", f"{conf_summary.get('final_score', 0.0):.2f}")
        c3.metric("Signals Resolved", signal_summary.get("shown", "—"))
        c4.metric("Validation",       "✅ Passed" if val_summary.get("passed") else "❌ Failed")

        st.divider()

        # Action log
        st.subheader("Action Log")
        action_log = exec_metrics.get("actions_applied", [])
        if action_log:
            log_df = pd.DataFrame(action_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No action log entries.")

        st.divider()

        # Violations
        violations = val_summary.get("violations", [])
        if violations:
            st.subheader("Accepted Warnings")
            st.caption("Non-critical violations that were accepted at max-iteration limit.")
            for v in violations:
                severity = v.get("severity", "unknown")
                message  = v.get("message", str(v))
                st.markdown(
                    f'<div class="violation-item">⚠ <strong>{severity}</strong> — {message}</div>',
                    unsafe_allow_html=True,
                )

        # Full JSON
        with st.expander("Full report JSON", expanded=False):
            st.json(report)