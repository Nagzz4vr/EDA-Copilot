"""
pages/5_Budget_Ledger.py  —  Token spend, cost breakdown, wall-time gauges,
phase bar chart, and raw JSONL ledger viewer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Budget — EDA Copilot", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.03em; }

    .gauge-container {
        background: #111827; border: 1px solid #1f2937; border-radius: 4px;
        padding: 14px 18px; margin-bottom: 8px;
    }
    .gauge-label { font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase;
                   color:#6b7280; margin-bottom:4px; }
    .gauge-value { font-size:1.4rem; font-family:'Syne',sans-serif; }
    .gauge-sub   { font-size:0.72rem; color:#6b7280; margin-top:2px; }
    .gauge-bar-outer { background:#1f2937; border-radius:2px; height:6px;
                       margin-top:8px; overflow:hidden; }
    .gauge-bar-inner { height:6px; border-radius:2px; transition:width 0.4s ease; }
    .bar-green  { background:#4ade80; }
    .bar-yellow { background:#facc15; }
    .bar-red    { background:#f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Guard ─────────────────────────────────────────────────────────────────
job_id = st.session_state.get("job_id")
if not job_id:
    st.warning("No active job — go to **Upload** first.")
    st.stop()

st.title("Budget & Token Ledger")
st.caption(f"Job `{job_id}`")

# ── Fetch budget data ─────────────────────────────────────────────────────
budget_controller = st.session_state.get("budget_controller")

usage:     Optional[Dict[str, Any]] = None
remaining: Optional[Dict[str, Any]] = None

if budget_controller:
    from utils.api_client import get_budget_usage, get_budget_remaining
    usage     = get_budget_usage(budget_controller, job_id)
    remaining = get_budget_remaining(budget_controller, job_id)
else:
    st.info(
        "`budget_controller` not found in `st.session_state`. "
        "Wire it up in your app factory to see live data."
    )

# ── Helper: render a progress gauge ──────────────────────────────────────
def _gauge(label: str, used: Any, total: Any, unit: str = "") -> None:
    try:
        pct = min(float(used) / float(total), 1.0) if float(total) > 0 else 0.0
    except (TypeError, ZeroDivisionError):
        pct = 0.0

    bar_class = "bar-green" if pct < 0.7 else ("bar-yellow" if pct < 0.9 else "bar-red")
    pct_display = f"{pct * 100:.1f}%"

    st.markdown(
        f"""
        <div class="gauge-container">
          <div class="gauge-label">{label}</div>
          <div class="gauge-value">{used}{unit}</div>
          <div class="gauge-sub">of {total}{unit} limit &nbsp;·&nbsp; {pct_display} used</div>
          <div class="gauge-bar-outer">
            <div class="gauge-bar-inner {bar_class}" style="width:{pct_display}"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Gauges row ────────────────────────────────────────────────────────────
if usage:
    limits = usage.get("limits", {})
    g1, g2, g3 = st.columns(3)

    with g1:
        _gauge(
            "Total Tokens",
            used=usage.get("total_tokens", 0),
            total=limits.get("max_tokens_total", 0),
        )

    with g2:
        _gauge(
            "Cost (USD)",
            used=usage.get("cost_usd", 0.0),
            total=limits.get("max_cost_usd", 0.0),
        )

    with g3:
        _gauge(
            "Wall Time",
            used=usage.get("elapsed_seconds", 0.0),
            total=limits.get("max_wall_seconds", 0.0),
            unit="s",
        )

    st.divider()

    # ── Phase breakdown bar chart ─────────────────────────────────────────
    st.subheader("Tokens by Phase")
    tokens_by_phase: Dict[str, int] = usage.get("tokens_by_phase", {})

    if tokens_by_phase:
        phase_df = pd.DataFrame(
            list(tokens_by_phase.items()),
            columns=["Phase", "Tokens"],
        ).sort_values("Tokens", ascending=False)
        st.bar_chart(phase_df.set_index("Phase")["Tokens"])
    else:
        st.caption("No phase breakdown available yet.")

    st.divider()

    # ── Remaining summary table ───────────────────────────────────────────
    if remaining:
        st.subheader("Remaining Budget")
        rem_df = pd.DataFrame(
            [
                {"Metric": "Tokens remaining",  "Value": f"{remaining.get('tokens_remaining', 0):,}"},
                {"Metric": "Cost remaining",    "Value": f"${remaining.get('cost_remaining', 0.0):.4f}"},
                {"Metric": "Seconds remaining", "Value": f"{remaining.get('seconds_remaining', 0):.0f}s"},
            ]
        )
        st.dataframe(rem_df, use_container_width=True, hide_index=True)

    # ── Configured limits ─────────────────────────────────────────────────
    with st.expander("Configured Limits", expanded=False):
        st.json(limits)

else:
    # No live budget controller — still show ledger if available
    st.caption("Live budget controller not available. Showing ledger data only.")

st.divider()

# ── Raw token ledger ──────────────────────────────────────────────────────
st.subheader("Raw Token Ledger")

from utils.api_client import get_ledger_rows

session_id  = st.session_state.get("session_id", job_id)
ledger_rows: List[Dict[str, Any]] = get_ledger_rows(session_id)

if ledger_rows:
    desired_cols = [
        "timestamp", "agent_id", "model",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cost", "latency_ms", "cache_hit", "retry_count", "tool_calls",
    ]
    ldf = pd.DataFrame(ledger_rows)
    display_cols = [c for c in desired_cols if c in ldf.columns]
    st.dataframe(ldf[display_cols], use_container_width=True, hide_index=True)

    # Totals footer
    numeric_cols = ["prompt_tokens", "completion_tokens", "total_tokens", "cost"]
    totals = {c: ldf[c].sum() for c in numeric_cols if c in ldf.columns}
    if totals:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Σ Prompt tokens",     f"{totals.get('prompt_tokens', 0):,}")
        t2.metric("Σ Completion tokens", f"{totals.get('completion_tokens', 0):,}")
        t3.metric("Σ Total tokens",      f"{totals.get('total_tokens', 0):,}")
        t4.metric("Σ Cost (USD)",         f"${totals.get('cost', 0.0):.4f}")

    with st.expander("Full JSONL entries", expanded=False):
        st.json(ledger_rows)
else:
    st.caption(f"No ledger entries found for session `{session_id}`.")
    st.caption("Entries are written by `TokenLedger.record()` during each LLM call.")