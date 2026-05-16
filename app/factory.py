from __future__ import annotations

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
import asyncio
import threading
from typing import Any, Dict

import logging
import traceback

import streamlit as st
 
# ── Observability ──────────────────────────────────────────────────────────
from observability.trace_logger import TraceLogger
from observability.step_sequencer import StepSequencer
from observability.confidence_tracker import ConfidenceTracker
from observability.anomaly_detector import AnomalyDetector
 
# ── Scoring ────────────────────────────────────────────────────────────────
from scoring.priority_scoring import PriorityScoring
from scoring.context_compresssor import ContextCompressor
 
# ── Cache ──────────────────────────────────────────────────────────────────
from core.cache.cache_manager import CacheManager
from core.cache.state_registry import StateRegistry
from core.cache.json_storage import JsonStorage
 
# ── Intelligence ───────────────────────────────────────────────────────────
from core.intelligence.signal_graph_builder import SignalGraphBuilder
from core.intelligence.plan_optimizer import PlanOptimizer
from core.intelligence.plan_refiner import PlanRefiner
from core.intelligence.global_state_simulator import GlobalStateSimulator
from core.hitl.hitl import HITL, RiskEngine
 
# ── Ingestion ──────────────────────────────────────────────────────────────
from core.ingestion import (
    Ingestor,
    Canonicalizer,
    ContextBuilder,
    TargetVariableSelector,
)

# ── Signals ────────────────────────────────────────────────────────────────
from signals.rule_engine import RuleEngine

# ── Validation ─────────────────────────────────────────────────────────────
from core.validation.validation_engine import ValidationEngine
from core.validation.sample_tester import SampleTester

# ── Execution ──────────────────────────────────────────────────────────────
from core.execution.dag_executor import DAGExecutor
from core.execution.dag_builder import DAGBuilder
from core.execution.state_manager import StateManager

# ── Output ─────────────────────────────────────────────────────────────────
from core.output.dataset_writer import DatasetWriter
from core.output.pipeline_exporter import PipelineExporter
from core.output.report_generator import ReportGenerator


# ── Agent ──────────────────────────────────────────────────────────────────
from core.agent.meta_agent import MetaAgent
from core.agent.agent import PlannerAgent

# ── HITL ───────────────────────────────────────────────────────────────────
from core.hitl.hitl_store import HitlStore
from core.hitl.hitl import HITLGate
# ── Budget ─────────────────────────────────────────────────────────────────
from control.budget_controller import *
from control.token_ledger import TokenLedger
from control.orchestrator import Orchestrator

# ── Config ─────────────────────────────────────────────────────────────────
from config import HITL_DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

bootstrap_logger = logging.getLogger("bootstrap")

# ── Stub — remove once DAG executor is complete ───────────────────────────
class NoOpTransformExecutor:
    def apply_single_transform(self, df, transform):
        return df
    

def _build_shared_infra() -> Dict[str, Any]:
    storage_backend  = JsonStorage(cache_dir="cache_storage")
    storage_provider = StateRegistry(storage=storage_backend)
    cache_manager    = CacheManager(
        storage_provider=storage_provider,
        current_rule_version="v1",
    )


    context_compressor = ContextCompressor()


    hitl_store = HitlStore(db_path=HITL_DB_PATH)
    hitl_gate  = HITLGate(hitl_store=hitl_store)


    budget_config = BudgetConfig(
        max_tokens_total    = 25_000,
        max_tokens_planning = 8_000,
        max_tokens_refining = 6_000,
        max_cost_usd        = 2.0,
        max_wall_seconds    = 1_000,
    )
    budget_controller = BudgetController(default_config=budget_config)

    return {
        "storage_backend":   storage_backend,
        "cache_manager":     cache_manager,
        "context_compressor": context_compressor,
        "hitl_store":        hitl_store,
        "hitl_gate":         hitl_gate,
        "budget_controller": budget_controller,
    }

def _make_agent_factory(shared: Dict[str, Any]):
    """
    Returns a closure. Orchestrator calls agent_factory(job_context)
    once per job; each call gets a fresh MetaAgent with job-scoped objects.
    """

    def agent_factory(job_context: Dict[str, Any]) -> MetaAgent:
        try:
            if job_context is None:
                raise ValueError("job_context is None")

            job_id = job_context["job_id"]

            bootstrap_logger.info(
                f"Creating MetaAgent for job_id={job_id}"
            )

            # ── Register job with budget controller ───────────────────────────
            shared["budget_controller"].register_job(job_id)

            # ── Job-scoped observability ──────────────────────────────────────
            logger             = TraceLogger(session_id=job_id)
            sequencer          = StepSequencer()
            confidence_tracker = ConfidenceTracker()
            anomaly_detector   = AnomalyDetector(
                confidence_tracker=confidence_tracker,
                logger=logger,
            )

            # ── Job-scoped scoring ────────────────────────────────────────────
            priority_scorer = PriorityScoring(context={})

            # ── Job-scoped validation ─────────────────────────────────────────
            transform_executor = NoOpTransformExecutor()
            sample_tester      = SampleTester(
                transform_executor=transform_executor,
                logger=logger,
            )
            validation_engine  = ValidationEngine(
                sample_tester=sample_tester,
                logger=logger,
            )

            # ── Job-scoped execution ──────────────────────────────────────────
            state_manager = StateManager(
                storage=shared["storage_backend"],
                logger=logger,
            )
            dag_executor = DAGExecutor(
                df=None,                   # MetaAgent assigns self.full_df later
                dag_builder=DAGBuilder,
                state_manager=state_manager,
                logger=logger,
            )
    
            # ── Job-scoped output ─────────────────────────────────────────────
            dataset_writer    = DatasetWriter()
            pipeline_exporter = PipelineExporter()
            report_generator  = ReportGenerator(
                confidence_tracker=confidence_tracker
            )

            # ── Job-scoped agent + reviewer ───────────────────────────────────
            planner_agent = PlannerAgent(session_id=job_id)

            # ── Job-scoped token ledger ───────────────────────────────────────
            token_ledger = TokenLedger(
                request_id=job_id,
                session_id=job_id,
            )
            # ── Assemble MetaAgent ────────────────────────────────────────────
            agent=MetaAgent(
                job_context=job_context,

                # ingestion — pass classes (MetaAgent instantiates per call)
                ingestor          = Ingestor,
                canonicalizer     = Canonicalizer,
                cache_manager     = shared["cache_manager"],
                context_builder   = ContextBuilder,
                variable_selector = TargetVariableSelector,

            # signals — pass class (MetaAgent instantiates with context)
                rule_engine = RuleEngine,

                # scoring
                priority_scorer    = priority_scorer,
                context_compressor = shared["context_compressor"],

                # intelligence
                signal_graph_builder   = SignalGraphBuilder(),
                plan_optimizer         = PlanOptimizer({}),
                global_state_simulator = GlobalStateSimulator,
                plan_refiner           = PlanRefiner,

                # agent
                agent    = planner_agent,
                reviewer = planner_agent,        # same instance; reviewer role is a method

                # risk engine — MetaAgent._review() uses this
                # replace with your real RiskEngine import when ready
                risk_engine = RiskEngine(logger=logger,),

                # validation
                validation_engine = validation_engine,

                # execution
                dag_executor  = dag_executor,
                state_manager = state_manager,

                # observability
                logger             = logger,
                step_sequencer     = sequencer,
                confidence_tracker = confidence_tracker,
                anomaly_detector   = anomaly_detector,

                # output
                dataset_writer    = dataset_writer,
                pipeline_exporter = pipeline_exporter,
                report_generator  = report_generator,

                # HITL
                hitl = shared["hitl_gate"],

                # budget
                budget_controller = shared["budget_controller"],
                token_ledger      = token_ledger,

                # iteration limits from job_context (set by SubmitRequest)
                max_iterations        = job_context.get("max_iterations", 3),
                max_review_iterations = 3,
            )
            return agent
        except Exception as e:
            bootstrap_logger.error(
                f"Failed to create MetaAgent: {e}"
            )
    
            bootstrap_logger.error(traceback.format_exc())
    
            raise

    return agent_factory

def _start_background_loop() -> asyncio.AbstractEventLoop:

    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True, name="pipeline-event-loop")
    t.start()
    return loop

def bootstrap() -> None:

    shared        = _build_shared_infra()
    agent_factory = _make_agent_factory(shared)

    orchestrator = Orchestrator(
        agent_factory       = agent_factory,
        max_concurrent_jobs = 4,
        job_timeout_seconds = 1_800,
    )


    bg_loop = _start_background_loop()
    future  = asyncio.run_coroutine_threadsafe(orchestrator.start(), bg_loop)
    future.result(timeout=10) 

    st.session_state.orchestrator       = orchestrator
    st.session_state.budget_controller  = shared["budget_controller"]
    st.session_state.hitl_store         = shared["hitl_store"]
    st.session_state.bg_loop            = bg_loop

