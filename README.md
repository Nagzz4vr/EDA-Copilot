# EDA Copilot — Autonomous Agentic Data Preparation System

An intelligent, multi-stage ML data preparation platform that autonomously profiles datasets, detects data quality issues, generates optimized preprocessing pipelines, validates transformation safety, and executes reproducible Directed Acyclic Graph (DAG) workflows. It features native execution observability, Human-in-the-Loop (HITL) governance, and budget-aware LLM orchestration.

Designed for production-grade preprocessing automation, **EDA Copilot** bridges the gap between rule-based statistical signal detection and LLM reasoning, incorporating graph optimization, safe state simulation, data validation, and rollback-safe execution into a unified orchestration framework.

---

## Overview

EDA Copilot accepts raw structured datasets (`CSV`, `Parquet`, `JSON`, `Excel`) and executes an automated, multi-agent pipeline:

* **Profile & Detect:** Extracts 100+ quality signals, statistical risks, and structural anomalies.
* **Graph Construction:** Builds deep dependency graphs of required cleaning and engineering operations.
* **LLM-Powered Planning:** Leverages Groq-hosted LLaMA models to formulate contextual data preparation steps.
* **Optimize & Simulate:** Performs a sandboxed dry-run simulation to project memory impacts, dimensionality growth, and data leakage without altering the core dataset.
* **Validate & Safeguard:** Tests transformations against baseline machine learning performance models.
* **Deterministic Execution:** Deploys tasks via a checkpointed, rollback-safe DAG execution engine.
* **Artifact Production:** Outputs pristine cleaned datasets, `scikit-learn`-compatible pipelines, and comprehensive compliance audit reports.

---

## Core System Architecture

The core framework is orchestrated around a deterministic Finite State Machine (FSM) controlled by a central coordinator, isolating individual domain boundaries:

```text
┌──────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                          │
│      Job queue • concurrency • timeout management            │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                        META AGENT                            │
│               Deterministic FSM Coordinator                  │
└──────────────────────────────────────────────────────────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      ▼                      ▼                      ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  INGESTION  │         │   SIGNALS   │         │ INTELLIGENCE│
│             │         │             │         │             │
│ • Ingestor  │────────>│ • RuleEngine│────────>│ • Planner   │
│ • Context   │         │ • Scorer    │         │ • Optimizer │
│ • Validator │         │ • Compressor│         │ • Refiner   │
└─────────────┘         └─────────────┘         └─────────────┘
                                                    │
                                 ┌──────────────────┘
                                 ▼
                       ┌────────────────────┐
                       │   REVIEW & HITL    │
                       │                    │
                       │ • Risk Engine      │
                       │ • HITL Gate        │
                       │ • LLM Reviewer     │
                       └────────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │     VALIDATION     │
                       │                    │
                       │ • Leakage Detector │
                       │ • Sample Tester    │
                       │ • Model Validator  │
                       └────────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │      EXECUTION     │
                       │                    │
                       │ • DAG Builder      │
                       │ • DAG Executor     │
                       │ • State Manager    │
                       └────────────────────┘
                                 │
                                 ▼
                       ┌────────────────────┐
                       │       OUTPUT       │
                       │                    │
                       │ • Dataset Writer   │
                       │ • Pipeline Export  │
                       │ • Report Generator │
                       └────────────────────┘

```

### Agent State Machine Lifecycle

The `MetaAgent` drives processing states sequentially to preserve predictability and state isolation:

```text
PERCEIVING ──▶ SIGNALING ──▶ MODELING ──▶ PLANNING ──▶ REVIEWING ──▶ OPTIMIZING
                                                                        │
COMPLETED  ◀── FINALIZING ◀── EXECUTING ◀── VALIDATING ◀── REFINING ◀── SIMULATING

```

---

## Core Features

### 1. Intelligent Signal Detection & Ranking

EDA Copilot runs over 100 deep rules across distinct validation domains:

* **Data Quality:** Automated missing-ness analysis, pattern mismatches, duplicate footprints, and structural entropy analysis.
* **Type Discovery:** Captures tricky edge cases like numeric-ID masquerading, constant/near-constant variance dropouts, and composite key structures.
* **Feature Engineering Routing:** Suggests targeted normalizations, high-cardinality aggregation boundaries, and structural text cleanings.
* **Feature Selection:** Evaluates multicollinearity matrices and clusters redundant features for proactive pruning.

To maximize context-window efficiency without blowing LLM tokens, detected signals are ranked via multi-factor prioritization scoring before compression:
`priority_score = severity × impact × context × urgency × decay`

### 2. Safety, Risk Assessment & HITL Governance

Operations are explicitly classified by structural risk categories (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`). High-risk mutations—such as `drop_column`, target-leaking operations, or aggressive `remove_outliers`—are explicitly put on hold, prompting human approval via an isolated Human-in-the-Loop (HITL) gateway.

### 3. Graph-Based Execution & Simulations

Transforms are completely untethered from fragile linear execution scripts. They are mapped explicitly to dependencies via an optimized DAG. Before modifications occur on the target volume, a sandboxed `GlobalStateSimulator` models data changes to forecast schema deviations and compute out-of-bounds metrics.

---

## Directory Structure

```text
EDA-copilot/
│
├── app/
│   ├── config.py                   # Env-var config (HITL DB path, poll interval, dirs)
│   ├── streamlit_app.py            # Root Streamlit entry point + global CSS + bootstrap
│   ├── factory.py                  # Wires all dependencies; builds MetaAgent per job
│   ├── pages/
│   │   ├── 1_upload.py             # File upload, schema preview, job submission
│   │   ├── 2_job_monitor.py        # Live state machine + confidence chart + trace log
│   │   ├── 3_hitl_review           # Human approval — action table, diff view, decision
│   │   ├── 4_results.py            # Dataset preview, pipeline .py, report JSON/Markdown
│   │   └── budget_ledger.py        # Token gauges, phase breakdown, raw JSONL ledger
│   └── utils/
│       └── api_client.py           # Thin async bridge — submit, status, HITL, artifacts
│
│
├── control/
│   ├── orchestrator.py             # Async job manager (semaphore, cancel, list)
│   ├── token_ledger.py             # Per-session JSONL token accounting
│   └── budget_controller.py        # Phase-level token + cost + wall-clock enforcement
│
├── core/
│   ├── __init__.py
│   ├── base_rule.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py                # PlannerAgent — LLM plan + review calls, fallback chain
│   │   ├── decision_schema.py      # AgentReview / ActionOverride Pydantic models
│   │   └── meta_agent.py           # MetaAgent — 13-state FSM driving the full pipeline
│   ├── cache/
│   │   ├── cache_manager.py        # Exact / partial hash lookup against state registry
│   │   ├── json_storage.py         # JSON-file storage backend
│   │   ├── state_registry.py       # UUID → record storage interface
│   │   └── storage_backend.py      # Abstract base for swappable storage backends
│   ├── execution/
│   │   ├── dag_builder.py          # Plan → topologically sorted DAG
│   │   ├── dag_executor.py         # Node-by-node execution with checkpoint rollback
│   │   ├── plan_normalizer.py      # Validates and type-repairs action list
│   │   ├── plan_translator.py      # OptimizedPlan → DAGBuilder-compatible dict
│   │   └── state_manager.py        # Persists / loads PlannerBundle post-execution
│   ├── hitl/
│   │   ├── __init__.py
│   │   ├── hitl.py                 # RiskEngine, HITLGate (async poll), HITL (CLI)
│   │   ├── hitl_store.py           # SQLite store for HITL request/decision rows
│   │   └── diff_viewer.py          # Before/after column-level stat delta
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── ingestor.py             # Streaming file loader with path-traversal guard
│   │   ├── context_builder.py      # Per-column stats: skew, entropy, outliers, patterns
│   │   ├── canonicalizer.py        # Deterministic hash-stable JSON representation
│   │   ├── data_validator.py       # Pydantic models for canonical output
│   │   └── target_variable_selector.py  # Schema-only column extraction (no full load)
│   ├── intelligence/
│   │   ├── __init__.py
│   │   ├── signal_graph_builder.py # Signals → ActionNode/ActionEdge graph
│   │   ├── plan_optimizer.py       # Greedy cost–benefit node selection + topo sort
│   │   ├── plan_refiner.py         # Post-simulation fixes: leakage, memory, OHE explosion
│   │   └── global_state_simulator.py    # Sample-based projection: leakage, memory, ordering
│   ├── output/
│   │   ├── dataset_writer.py       # Writes Parquet / CSV / JSON + WriteManifest
│   │   ├── pipeline_exporter.py    # sklearn Pipeline .pkl + .py reconstruction script
│   │   └── report_generator.py     # JSON + Markdown decision report
│   ├── routing/
│   │   └── confidence_router.py    # Classifies signal blob → HIGH / MEDIUM / LOW tier
│   ├── schema/
│   │   ├── __init__.py
│   │   ├── planner_schema.py       # OptimizedPlan, SignalGraph, PlannerBundle, etc.
│   │   └── shared_shema.py         # ExecutionPlan, PlanMetadata dataclasses
│   └── validation/
│       ├── leakage.py              # Spearman / point-biserial leakage scorer
│       ├── sample_tester.py        # Applies transforms on sample, collects violations
│       └── validation_engine.py    # DummyClassifier before/after delta check
│
├── observability/
│   ├── __init__.py
│   ├── anomaly_detector.py         # Drop / floor / sustained-low anomaly checks
│   ├── confidence_tracker.py       # Per-UUID score history
│   ├── step_sequencer.py           # State transition timeline
│   └── trace_logger.py             # JSONL trace per session (every tool call)
│
├── scoring/
│   ├── __init__.py
│   ├── context_compresssor.py      # Token-budget trimming of ranked signal list
│   └── priority_scoring.py         # severity × impact × type → 0-100 score
│
└── signals/
    ├── __init__.py
    ├── rule_engine.py              # Auto-discovers BaseRule subclasses, runs in priority order
    └── rules/
        ├── __init__.py
        ├── base_rule.py            # BaseRule ABC + RuleOutput dataclass
        ├── analysis/
        │   ├── __init__.py
        │   └── archetype_detection.py   # TimeSeries, Transactional, IoT, NLP dataset rules
        ├── column_detection/
        │   ├── __init__.py
        │   ├── col_type.py              # Numeric-ID masquerade, low-card numeric, high-card cat
        │   ├── id_detection.py          # Definite ID, composite key candidate
        │   └── low_variance.py          # Constant column, near-constant numeric
        ├── data_quality/
        │   ├── __init__.py
        │   ├── deduplication.py         # Critical / high / low / no-duplicate rules
        │   └── missing_data.py          # High / moderate / pattern / random missing
        ├── feature_engineering/
        │   ├── __init__.py
        │   ├── encoding.py              # Binary label, low-card OHE, imbalanced grouping
        │   ├── normalization.py         # Highly skewed numeric → log transform
        │   └── text_processing.py       # Long text NLP routing, short text → categorical
        ├── feature_selection/
        │   ├── __init__.py
        │   └── multicollinearity.py     # High / moderate correlation, cluster redundancy
        └── visualization/
            ├── __init__.py
            └── plot_selection.py        # Distribution, log, bar, heatmap plot triggers

```

---

## Supported Operations

| Strategy Type | Supported Transformation Operations |
| --- | --- |
| **Imputation** | Mean, Median, Mode, Constant Replacement, K-Nearest Neighbors (KNN) |
| **Encoding** | One-Hot Encoding, Ordinal Encoding, Target (Bayesian) Encoding |
| **Scaling** | Standard Scaling, Min-Max Uniform Scaling, Outlier-Robust Scaling |
| **Structural Cleaning** | Row Deduplication, Outlier Clipping, Explicit Casting, Text Standardizations, Structural Dropping |

---

## Streamlit UI Ecosystem

The system includes a production-ready dashboard layout mapping runtime components directly to monitoring surfaces:

* **Upload Portal:** Visual Schema analysis, high-level dataset profiling, and multi-mode target boundary definitions.
* **Job Monitor:** Real-time state-machine updates, execution path updates, and active confidence trendlines.
* **HITL Gateway Review:** Clear, isolated differential summaries showing statistical adjustments alongside granular Approve/Reject interactions.
* **Result Matrix:** Direct pipeline deployment exports (`scikit-learn` artifacts) paired with transformed table download channels.
* **Budget Ledger Control:** Complete token telemetry tracking input/output costs, resource usage caps, and JSONL log traces.

---

## Installation & Setup

### 1. Build Environment Block

```bash
# Set up isolated conda workspace Environment
conda create -n eda python=3.10 -y
conda activate eda

# Install production and agent runtimes
pip install -r requirements.txt

```

### 2. Environment Configurations

Clone the distribution environment template configuration to activate processing keys:

```bash
cp .env.example .env

```

Populate your configuration values in the generated `.env` file:

```env
GROQ_API_KEY=your_groq_llm_api_key_here
HITL_DB_PATH=core/hitl/hitl_governance.db
HITL_POLL_INTERVAL=2.0
HITL_TIMEOUT_SECONDS=3600
OUTPUT_DIR=outputs
LEDGER_DIR=control/ledgers

```

---

## Quickstart Guide

### Option A: Launching the Interactive Interface

To start the multi-page dashboard, execute the following command:

```bash
streamlit run app/streamlit_app.py

```

### Option B: Programmatic Driver Orchestration

For end-to-end processing within automation scripts or data processing pipelines:

```python
import asyncio
from control.orchestrator import Orchestrator, SubmitRequest
from app.factory import build_agent 

async def main():
    # Instantiate the coordinator container engine
    orchestrator = Orchestrator(agent_factory=build_agent)
    await orchestrator.start()

    # Formulate configuration constraints 
    request = SubmitRequest(
        file_path="data/titanic.csv",
        base_dir=".",
        target_column="Survived",
        max_iterations=3,
        budget_tokens=25000,
    )

    # Dispatches asynchronously down through the FSM State Matrix
    job_status = await orchestrator.submit_job(request)
    print(f"Job processing established successfully. ID Reference: {job_status.job_id}")

if __name__ == "__main__":
    asyncio.run(main())

```

---

## Known System Boundaries

* **Simulations:** The `GlobalStateSimulator.apply_single_action` strategy contains isolated layout stubs for complex custom operations.
* **Validations:** Validation modules use an internal `DummyClassifier` implementation baseline for scoring integrity evaluations.
* **Concurrences:** The underlying SQLite-backed storage configuration handling HITL processing blocks is designed for single-node executions.

---

## Future Development Path

* Distributed execution using Ray and Polars backends.
* High-throughput HITL event queues with Redis and PostgreSQL integration.
* Ensemble planning models using cross-agent evaluation schemes.
* Advanced semantic query layer caching for high-speed processing replays.
* Turnkey deployment setups optimized for native cloud runtime environments.

---

## License

Distributed under the terms of the **MIT License**. For details, see `LICENSE.txt`.
