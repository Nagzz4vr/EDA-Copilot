from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
from core.ingestion.target_variable_selector import TargetVariableSelector
from control.orchestrator import Orchestrator
from core.ingestion.ingestor import Ingestor


st.set_page_config(page_title="Upload — EDA Copilot", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500&family=Syne:wght@700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.03em; }
    .stButton > button { font-family: 'DM Mono', monospace; border-radius: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)
#upload
st.title("Upload Dataset")
st.caption("Accepts CSV · Parquet · JSON · Excel")

uploaded = st.file_uploader(
    "Drop your dataset here",
    type=["csv", "parquet", "json","xlsx"],
    label_visibility="collapsed",
)


if not uploaded:
    st.info("Upload a file to continue.")
    st.stop()


temp_dir = tempfile.mkdtemp()
temp_path = Path(temp_dir) / uploaded.name
with open(temp_path, "wb") as f:
    f.write(uploaded.getbuffer())

try:
    ingestor = Ingestor(
        filepath=str(temp_path),
        base_dir=temp_dir,
    )

    selector = TargetVariableSelector(
        filepath=str(temp_path),
        base_dir=temp_dir,
    )

except Exception as e:
    st.error(f"Failed to initialize dataset: {e}")
    st.stop()


try:
    columns = selector.load_column_names()

except Exception as e:
    st.error(f"Failed to read schema: {e}")
    st.stop()

raw_bytes = uploaded.read()

df_preview = ingestor.head(50)

col_left, col_right = st.columns([2, 1])



with col_left:
    st.subheader("Preview (first 50 rows)")
    st.dataframe(
        df_preview.head(20),
        use_container_width=True,
    )

with col_right:

    st.subheader("Schema")

    schema_df = pd.DataFrame(
        {
            "column": df_preview.columns,
            "dtype": df_preview.dtypes.astype(str).values,
        }
    )

    st.dataframe(
        schema_df,
        use_container_width=True,
        hide_index=True,
    )
st.divider()

st.subheader("Configuration")

cfg_col1, cfg_col2 = st.columns(2)

with cfg_col1:

    target_col = st.selectbox(
        "Target column",
        options=columns,
        index=len(columns) - 1 if columns else 0,
        help=(
            "The column the model will predict. "
            "Never transformed except imputation."
        ),
    )

    max_iterations = st.slider(
        "Max refinement iterations",
        min_value=1,
        max_value=5,
        value=3,
        help=(
            "How many SIMULATING → REFINING loops "
            "before forcing VALIDATING."
        ),
    )

with cfg_col2:

    with st.expander("Budget limits", expanded=False):

        max_tokens = st.number_input(
            "Max total tokens",
            value=25_000,
            step=1_000,
        )

        max_cost_usd = st.number_input(
            "Max cost (USD)",
            value=2.0,
            step=0.50,
        )

        max_wall_sec = st.number_input(
            "Max wall time (s)",
            value=1_000,
            step=100,
        )

        plan_tokens = st.number_input(
            "Planning phase token cap",
            value=8_000,
            step=500,
        )

        refine_tokens = st.number_input(
            "Refining phase token cap",
            value=6_000,
            step=500,
        )


budget_overrides = {
    "max_tokens_total": int(max_tokens),
    "max_cost_usd": float(max_cost_usd),
    "max_wall_seconds": int(max_wall_sec),
    "max_tokens_planning": int(plan_tokens),
    "max_tokens_refining": int(refine_tokens),
}
st.divider()

submit = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

if submit:
    # Save the uploaded file to a temp path that persists for the job lifetime
    local_tmp_dir = Path(ROOT_DIR) / "tmp"
    local_tmp_dir.mkdir(exist_ok=True)

    suffix = Path(uploaded.name).suffix

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=local_tmp_dir)
    tmp.write(raw_bytes)
    tmp.flush()
    tmp.close()
    file_path = tmp.name

    # Retrieve (or lazily create) the orchestrator from session state
    if "orchestrator" not in st.session_state:

        st.session_state.orchestrator = Orchestrator()

    orchestrator = st.session_state.orchestrator
    if orchestrator is None:
        st.error(
            "Orchestrator not initialised in `st.session_state.orchestrator`. "
            "Wire it up in your app factory / entrypoint before navigating here."
        )
        st.stop()

    from app.utils.api_client import submit_job

    with st.spinner("Submitting job…"):
        try:
            job_id = submit_job(
    orchestrator=orchestrator,
    file_path=file_path,
    base_dir=str(local_tmp_dir),
    target_col=target_col,
    budget_overrides=budget_overrides,
    max_iterations=max_iterations,
)
        except Exception as exc:
            st.error(f"Submission failed: {exc}")
            st.stop()

    st.session_state.job_id      = job_id
    st.session_state.session_id  = job_id   
    st.session_state.job_status  = None
    st.session_state.hitl_pending = False
    st.session_state.artifacts   = None

    st.success(f"Job submitted — `{job_id}`")
    st.switch_page(r"pages\2_job_monitor.py")
    st.info("Navigate to **Job Monitor** to track progress.")