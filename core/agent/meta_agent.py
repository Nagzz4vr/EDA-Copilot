from enum import Enum, auto
from typing import Dict, Optional, Any
import pandas as pd
import hashlib
import asyncio
import time
import traceback
from copy import deepcopy
from core.ingestion.data_validator import CanonicalizedOutput
from core.execution.plan_translator import PlanTranslator
from core.schema.planner_schema import *
from core.execution.plan_normalizer import PlanNormalizer

class State(Enum):

    PERCEIVING = auto()
    SIGNALING  = auto()
    MODELING   = auto()

    PLANNING   = auto()
    REVIEWING  = auto()
    OPTIMIZING = auto()

    SIMULATING = auto()
    REFINING   = auto()
    VALIDATING = auto()

    EXECUTING  = auto()
    FINALIZING = auto()
    COMPLETED  = auto()

    FAILED     = auto()


class MetaAgent:
    def __init__(self, job_context, **kwargs):
        self.context    = job_context
        self.state      = State.PERCEIVING
        self.prev_state: Optional[State] = None
        self.max_iterations  = kwargs.get("max_iterations", 3)
        self.iteration_count = 0

        self.ingestor          = kwargs["ingestor"]
        self.canonicalizer     = kwargs["canonicalizer"]
        self.cache_manager     = kwargs["cache_manager"]
        self.context_builder   = kwargs["context_builder"]
        self.variable_selector = kwargs["variable_selector"]

        self.rule_engine        = kwargs["rule_engine"]
        self.priority_scorer    = kwargs["priority_scorer"]
        self.context_compressor = kwargs["context_compressor"]
        self.agent              = kwargs["agent"]  # LLM planner
        self.hitl               = kwargs.get("hitl")

        self.graph_builder        = kwargs["signal_graph_builder"]
        self.optimizer            = kwargs["plan_optimizer"]
        self.GlobalStateSimulator = kwargs["global_state_simulator"]
        self.refiner              = kwargs["plan_refiner"]

        self.confidence_tracker = kwargs["confidence_tracker"]
        self.anomaly_detector   = kwargs["anomaly_detector"]
        self.validator          = kwargs["validation_engine"]

        self.executor          = kwargs["dag_executor"]
        self.logger            = kwargs["logger"]
        self.step_sequencer    = kwargs["step_sequencer"]
        self.dataset_writer    = kwargs["dataset_writer"]
        self.pipeline_exporter = kwargs["pipeline_exporter"]
        self.report_generator  = kwargs["report_generator"]
        self.state_manager     = kwargs["state_manager"]
        self.budget_controller  = kwargs.get("budget_controller")
        self.token_ledger       = kwargs.get("token_ledger")


        self.optimized_plan:   Optional[OptimizedPlan]   = None
        self.signal_graph:     Optional[SignalGraph]      = None
        self.validation_report: Optional[ValidationReport] = None

        self.plan:             Optional[Any] = None
        self.signal_bag:       Optional[Any] = None
        self.signal_graph:     Optional[Any] = None
        self.optimized_plan:   Optional[Any] = None
        self.impact_report:    Optional[Any] = None
        self.canonical_output: Optional[Any] = None
        self.state_uuid:       Optional[str] = None
        self.confidence_score: float = 0.0


        self.execution_result:  Optional[Any] = None
        self.cache_hit: bool = False



    async def run(self):
        try:
            while self.state not in [State.COMPLETED, State.FAILED]:

                if self.state == State.PERCEIVING:
                    self._transition(State.PERCEIVING, trigger="AGENT_START")
                    self.state = await self._perceive()

                elif self.state == State.SIGNALING:
                    self._transition(State.SIGNALING, trigger="CACHE_MISS")
                    self.state = await self._signal()

                elif self.state == State.MODELING:
                    self._transition(State.MODELING, trigger="SIGNALS_PROCESSED")
                    self.state = await self._model()

                elif self.state == State.PLANNING:
                    self._transition(State.PLANNING, trigger="GRAPH_COMPILED")
                    self.state = await self._plan()

                elif self.state == State.OPTIMIZING:
                    self._transition(State.OPTIMIZING, trigger="PLAN_GENERATED")
                    self.state = await self._optimize()

                elif self.state == State.SIMULATING:
                    self._transition(State.SIMULATING, trigger="PLAN_OPTIMIZED")
                    self.state = await self._simulate()

                elif self.state == State.REFINING:
                    self._transition(State.REFINING, trigger="SIMULATION_COMPLETE")
                    self.state = await self._refine()

                elif self.state == State.REVIEWING:
                    self._transition(State.REVIEWING, trigger="PLAN_VIOLATION_DETECTED")
                    self.state = await self._review()

                elif self.state == State.VALIDATING:
                    self._transition(State.VALIDATING, trigger="CONVERGENCE_REACHED")
                    self.state = await self._validate()

                elif self.state == State.EXECUTING:
                    self._transition(State.EXECUTING, trigger="VALIDATION_PASSED")
                    self.state = await self._execute()

                elif self.state == State.FINALIZING:
                    self._transition(State.FINALIZING, trigger="EXECUTION_COMPLETE")
                    self.state = await self._finalize()

                else:
                    print(f"[DEV] No handler for state {self.state.name} — stopping early")
                    break

        except Exception as e:
            prev       = self.state
            self.state = State.FAILED
            self._transition(State.FAILED, trigger="CONTRACT_VIOLATION")
            print(f"FATAL ERROR in state {prev.name}: {e}")
            traceback.print_exc()

        print(f"Job {self.context['job_id']} finished with state: {self.state.name}")


    async def _perceive(self) -> State:
        self.logger.log(tool="INGESTOR", intent="Starting perception phase",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            selector   = self.variable_selector(self.context["file_path"])
            columns    = await asyncio.to_thread(selector.load_column_names)
            target_col = self.context.get("target_column") or columns[-1]

            ingestor = self.ingestor(self.context["file_path"])
            batches  = []
            async for batch in self._iter_batches(ingestor, limit=50_000):
                batches.append(batch)
            full_df      = pd.concat(batches, ignore_index=True)
            self.full_df = full_df.copy()

            builder      = self.context_builder(full_df, target_col=target_col)
            raw_analysis = await asyncio.to_thread(builder.build_context)

            def _canonicalize():
                c = self.canonicalizer(raw_analysis)
                return c.add_state_info()

            raw_canonical         = await asyncio.to_thread(_canonicalize)
            validated_output      = CanonicalizedOutput(**raw_canonical)
            self.canonical_output = validated_output
            self.state_uuid       = validated_output.metadata.state_uuid

            self.dataset_sample = full_df.head(500)
            self.dataset_schema = {
                "dataset_overview": {
                    "num_rows":    len(full_df),
                    "num_columns": len(full_df.columns),
                },
                "label_column": target_col,
            }

            cache_package = {
                "metadata": {
                    "state_uuid":  self.canonical_output.metadata.state_uuid,
                    "fingerprint": self.canonical_output.metadata.fingerprint,
                }
            }

            hit_type, matched_uuid = await asyncio.to_thread(
                self.cache_manager.lookup, cache_package
            )


            self.logger.log(
                tool="CACHE_MANAGER",
                intent=f"Cache lookup → {hit_type}",
                inputs={"state_uuid": self.state_uuid},
                outputs={"hit_type": hit_type, "matched_uuid": matched_uuid},
                confidence=1.0,
            )


            if hit_type in ("EXACT_HIT", "PARTIAL_HIT") and matched_uuid:
                record = await asyncio.to_thread(
                    self.state_manager.load, matched_uuid
                )

                if record and record.get("planner_bundle"):
                    bundle = PlannerBundle.model_validate(record["planner_bundle"])


                    self.optimized_plan = bundle.optimized_plan
                    self.signal_graph   = bundle.signal_graph
                    self.cache_hit      = True

                    self.logger.log(
                        tool="CACHE_MANAGER",
                        intent=f"{hit_type} — PlannerBundle restored, routing to EXECUTING",
                        inputs={"matched_uuid": matched_uuid},
                        outputs={
                            "action_count":   len(self.optimized_plan.actions),
                            "signal_nodes":   len(self.signal_graph.nodes),
                        },
                        confidence=1.0,
                    )

                    return State.EXECUTING

                self.logger.log(
                    tool="CACHE_MANAGER",
                    intent=f"{hit_type} returned uuid but no bundle — falling through to full pipeline",
                    inputs={"matched_uuid": matched_uuid},
                    outputs={},
                    confidence=0.5,
                )


            return State.SIGNALING

        except Exception as exc:
            self.logger.log(tool="PERCEPTION_ERROR", intent="Perception failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

    async def _signal(self) -> State:
        self.logger.log(tool="RULE_ENGINE", intent="Starting signaling phase",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            context = self.canonical_output.canonical_data.model_dump()
            context.setdefault("dataset_health", {
                "duplicate_percent": 0,
                "missing_percent":   0,
            })
            context.setdefault("top_correlations", [])

            engine      = self.rule_engine(context=context, logger=self.logger)
            raw_signals = await asyncio.to_thread(engine.run)

            scored_signals  = await asyncio.to_thread(
                self.priority_scorer.rank_signals, raw_signals
            )
            self.signal_bag = await asyncio.to_thread(
                self.context_compressor.compress, scored_signals
            )

            post_count = len(self.signal_bag.signals) if self.signal_bag else 0
            self.logger.log(tool="RULE_ENGINE", intent="Signals scored and compressed",
                            inputs={}, outputs={"signal_count": post_count},
                            confidence=1.0)

        except Exception as exc:
            self.logger.log(tool="SIGNALING_ERROR", intent="Signaling failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

        return State.MODELING

    async def _model(self) -> State:
        self.logger.log(tool="SIGNAL_GRAPH_BUILDER", intent="Starting graph building",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            raw_graph = await asyncio.to_thread(
                self.graph_builder._build_graph, self.signal_bag
            )

            self.signal_graph = self._coerce_to_signal_graph(raw_graph)

            if not self.signal_graph or not self.signal_graph.nodes:
                self.logger.log(tool="SIGNAL_GRAPH_BUILDER",
                                intent="Empty graph — no actionable nodes produced",
                                inputs={}, outputs={}, confidence=0.5)

        except Exception as exc:
            self.logger.log(tool="MODELING_ERROR", intent="Modeling phase failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

        return State.PLANNING

    async def _plan(self) -> State:

        self.logger.log(
            tool="PLAN",
            intent="Planning phase start — LLM call",
            inputs={}, outputs={}, confidence=1.0
        )

        try:

            if self.budget_controller:
                await asyncio.to_thread(
                    self.budget_controller.assert_budget_available,
                    phase="PLANNING"
                )

            plan_result = await self.agent.generate_plan(
                signal_graph=self.signal_graph,
                canonical_context=self.canonical_output,
            )


            if self.token_ledger:
                await asyncio.to_thread(
                    self.token_ledger.record,
                    phase="PLANNING",
                    tokens_used=plan_result.tokens_used,
                )

            self.decision_plan = DecisionPlan(
                actions=plan_result.actions,
                confidence_score=plan_result.confidence_score,
                reasoning=getattr(plan_result, "reasoning", ""),
                metadata=getattr(plan_result, "metadata", {}),
            )
            
            self.confidence_score = plan_result.confidence_score


            await asyncio.to_thread(
                self.confidence_tracker.record,
                phase="PLANNING",
                score=self.confidence_score,
            )

            self.logger.log(
                tool="PLAN",
                intent="Plan generated",
                inputs={}, 
                outputs={
                    "action_count": len(self.decision_plan.actions),
                    "confidence": self.confidence_score,
                },
                confidence=self.confidence_score
            )

        except Exception as exc:
            self.logger.log(
                tool="PLAN_ERROR",
                intent="Planning failed",
                inputs={}, 
                outputs={"error": str(exc)},
                confidence=0.0
            )
            raise
        
        return State.REVIEWING
    
    async def _review(self) -> State:
        self.logger.log(
            tool="REVIEW",
            intent="Review phase start",
            inputs={},
            outputs={},
            confidence=1.0
        )

        try:
            # ─────────────────────────────────────────
            # 1. Resolve plan safely (single source)
            # ─────────────────────────────────────────
            plan = self.plan
            if plan is None:
                raise RuntimeError("No plan available for review phase")

            plan_dict = (
                plan.model_dump()
                if hasattr(plan, "model_dump")
                else deepcopy(plan)
            )

            risk_result = self.risk_engine.assess(
                plan=plan_dict,
                metadata={"target_column": self.context.get("target_column")}
            )

      
            if risk_result["requires_hitl"]:
                if not self.hitl:
                    raise RuntimeError("HITL required but not configured")

                action = await self.hitl.request_approval(
                    plan=plan_dict,
                    risk_result=risk_result,
                    state_uuid=self.state_uuid,
                )

                if action == "REJECT":
                    self.logger.log(
                        tool="HITL",
                        intent="Human rejected plan",
                        inputs={"state_uuid": self.state_uuid},
                        outputs={},
                        confidence=1.0
                    )
                    return State.REFINING

                self.logger.log(
                    tool="HITL",
                    intent="Human approved plan",
                    inputs={"state_uuid": self.state_uuid},
                    outputs={"action": action},
                    confidence=1.0
                )


            review_result = self.reviewer.review_plan(
                impact_report=self.impact_report,
                optimized_plan=plan_dict
            )

            if not review_result.approved:
                self.logger.log(
                    tool="REVIEW",
                    intent="LLM rejected plan",
                    inputs={},
                    outputs=review_result.model_dump(),
                    confidence=0.3
                )
                return State.REFINING


            if review_result.overrides:
                plan_dict = self._apply_overrides(plan_dict, review_result.overrides)

            normalizer = PlanNormalizer(
                canonical_context=self.canonical_output.canonical_data.model_dump()
            )

            normalized_plan = normalizer.normalize(plan_dict)

   
            self.plan = normalized_plan

            self.logger.log(
                tool="REVIEW",
                intent="Review completed successfully",
                inputs={},
                outputs={
                    "risk_level": risk_result["risk_level"],
                    "hitl_required": risk_result["requires_hitl"],
                    "approved": True
                },
                confidence=1.0
            )

            return State.OPTIMIZING

        except Exception as exc:
            self.logger.log(
                tool="REVIEW_ERROR",
                intent="Review failed",
                inputs={},
                outputs={"error": str(exc)},
                confidence=0.0
            )
            raise


    async def _optimize(self) -> State:
        self.logger.log(tool="PLAN_OPTIMIZER", intent="Starting optimization",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            raw_plan = await asyncio.to_thread(
                self.optimizer.optimize, self.signal_graph.model_dump()
            )

            self.optimized_plan = self._coerce_to_optimized_plan(raw_plan)
 
            self.logger.log(
                tool="PLAN_OPTIMIZER",
                intent="Optimization complete",
                inputs={},
                outputs={"action_count": len(self.optimized_plan.actions)},
                confidence=1.0,
            )
        except Exception as exc:
            self.logger.log(tool="OPTIMIZING_ERROR", intent="Optimization phase failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

        return State.SIMULATING

    async def _simulate(self) -> State:
        self.logger.log(tool="SIMULATOR", intent="Simulation phase start",
                        inputs={}, outputs={}, confidence=1.0)
        try:
    
            plan_dict = self.optimized_plan.model_dump()
            sim = self.GlobalStateSimulator(
                plan=plan_dict,
                dataset_sample=self.dataset_sample,
                dataset_schema=self.dataset_schema,
            )
            self.impact_report = await asyncio.to_thread(sim.simulate)

            anomalies = await asyncio.to_thread(
                self.anomaly_detector.check, self.state_uuid
            )
            if anomalies:
                self.logger.log(
                    tool="ANOMALY_DETECTOR",
                    intent="Anomalies detected — routing to HITL review",
                    inputs={}, outputs={"anomalies": anomalies},
                    confidence=0.5,
                )
                return State.REVIEWING

        except Exception as exc:
            self.logger.log(tool="SIMULATING_ERROR", intent="Simulation phase failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

        return State.REFINING

    async def _refine(self) -> State:
        self.logger.log(tool="PLAN_REFINER", intent="Refinement phase start",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            refiner_instance = self.refiner(
                plan=self.optimized_plan.model_dump(),
                impact_report=self.impact_report,
                signal_graph=self.signal_graph.model_dump(),
                label_column=self.dataset_schema.get("label_column"),
                dataset_sample=self.dataset_sample,
            )

            raw_plan = await asyncio.to_thread(refiner_instance.refine)
            self.optimized_plan  = self._coerce_to_optimized_plan(raw_plan)
            self.iteration_count += 1

            await asyncio.to_thread(
                self.confidence_tracker.update,
                state_uuid=self.state_uuid,
                step=f"REFINING_{self.iteration_count}",
                score=self.confidence_score,
            )

            is_anomaly = await asyncio.to_thread(
                self.anomaly_detector.check, self.state_uuid
            )

            if is_anomaly or self.iteration_count >= self.max_iterations:
                return State.VALIDATING

        except Exception as exc:
            self.logger.log(tool="REFINING_ERROR", intent="Refinement phase failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

        return State.SIMULATING

    

    async def _validate(self) -> State:
        self.logger.log(tool="VALIDATOR", intent="Validation phase start",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            transforms = [a.model_dump() for a in self.optimized_plan.actions]

            raw_report = await asyncio.to_thread(
                self.validator.run,
                self.dataset_sample,
                self.context["target_column"],
                transforms,
            )

            self.validation_report = self._coerce_to_validation_report(raw_report)

            if self.validation_report.passed:
                return State.EXECUTING

            has_critical = any(
                v.get("severity") == "CRITICAL"
                for v in self.validation_report.violations
            )

            if has_critical:
                raise RuntimeError(
                    f"Validation failed — critical violations: "
                    f"{[v for v in self.validation_report.violations if v.get('severity') == 'CRITICAL']}"
                )

            iterations_left = self.iteration_count < self.max_iterations

            if iterations_left:
                self.logger.log(
                    tool="VALIDATOR",
                    intent="Non-critical violations detected — refining",
                    inputs={},
                    outputs={"violations": self.validation_report.violations},
                    confidence=0.6
                )
                return State.REFINING

            # Max iterations reached with only non-critical violations
            self.logger.log(
                tool="VALIDATOR",
                intent="Max iterations reached — proceeding with warnings",
                inputs={},
                outputs={"violations": self.validation_report.violations},
                confidence=0.5
            )
            return State.EXECUTING

        except RuntimeError:
            raise
        except Exception as exc:
            self.logger.log(tool="VALIDATION_ERROR",
                            intent="Validation unexpected error",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

    async def _execute(self) -> State:
        self.logger.log(tool="DAG_EXECUTOR", intent="Execution phase start",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            # ── HARD INVARIANTS ────────────────────────────────────────────
            if not isinstance(self.optimized_plan, OptimizedPlan):
                raise RuntimeError(
                    f"optimized_plan must be OptimizedPlan, "
                    f"got {type(self.optimized_plan)!r}. "
                    "A coercion step was skipped upstream."
                )
            if not self.optimized_plan.actions:
                raise RuntimeError(
                    "optimized_plan has no actions — cannot build a DAG. "
                    "Planner or cache-restore produced an empty plan."
                )
 
            # ── BUILD VALIDATED EXECUTION INPUT ────────────────────────────
            execution_input = ExecutionInput(
                optimized_plan=self.optimized_plan,
                signal_graph=self.signal_graph or SignalGraph(nodes=[], edges=[]),
            )
 
            self.executor.df = self.full_df
 
            translator = PlanTranslator(
    refined_plan=self.optimized_plan.model_dump()
    if hasattr(self.optimized_plan, "model_dump")
    else self.optimized_plan.__dict__,
    signal_graph=execution_input.signal_graph,
)
            execution_plan = translator.translate()
 
            self.execution_result = await self.executor.run(
                plan=execution_plan,
                mode="FINAL",
                state_uuid=self.state_uuid,
            )
 
        except Exception as exc:
            self.logger.log(tool="EXECUTION_ERROR", intent="Execution failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise
 
        return State.FINALIZING
 
    async def _finalize(self) -> State:
        self.logger.log(tool="FINALIZER", intent="Finalization phase start",
                        inputs={}, outputs={}, confidence=1.0)
        try:
            if self.execution_result is None:
                raise RuntimeError(
                    "execution_result is None in FINALIZING — "
                    "_execute must complete successfully before _finalize runs."
                )
 
            # ── write dataset ──────────────────────────────────────────────
            dataset_manifest = await asyncio.to_thread(
                self.dataset_writer.write,
                self.execution_result,
                self.context["job_id"],
            )
 
            # ── export sklearn pipeline ────────────────────────────────────
            pipeline_manifest = await asyncio.to_thread(
                self.pipeline_exporter.export,
                self.optimized_plan.model_dump(),
                self.context["job_id"],
            )
 
            # ── generate decision report ───────────────────────────────────
            report_path = await asyncio.to_thread(
                self.report_generator.generate,
                self.optimized_plan.model_dump(),
                self.execution_result,
                self.signal_bag.signals if self.signal_bag else [],
                self.confidence_score,
                self.context["job_id"],
                self.validation_report.model_dump() if self.validation_report else None,
            )
 
            # ── persist PlannerBundle — SINGLE SOURCE OF TRUTH ────────────
            # Rule: only PlannerBundle crosses the persistence boundary.
            # DataFrames live exclusively in DatasetWriter output files.
            bundle = PlannerBundle(
                signal_graph=self.signal_graph or SignalGraph(nodes=[], edges=[]),
                optimized_plan=self.optimized_plan,
                validation_report=self.validation_report,
                fingerprint=self.canonical_output.metadata.fingerprint,
                state_uuid=self.state_uuid,
            )

            await asyncio.to_thread(
                self.state_manager.save,
                state_uuid=self.state_uuid,
                planner_bundle=bundle,
                result={
                    "rows_processed":  self.execution_result.get("rows_processed"),
                    "actions_applied": self.execution_result.get("actions_applied", []),
                    "dag_version":     self.execution_result.get("dag_version"),
                    "errors":          self.execution_result.get("errors", []),
                },
            )

            self.final_artifacts = {
                "dataset":  dataset_manifest.to_dict(),
                "pipeline": pipeline_manifest.to_dict(),
                "report":   report_path,
            }

            self.logger.log(tool="FINALIZER",
                            intent="Artifacts generated and bundle persisted",
                            inputs={}, outputs=self.final_artifacts, confidence=1.0)

        except Exception as exc:
            self.logger.log(tool="FINALIZATION_ERROR", intent="Finalization failed",
                            inputs={}, outputs={"error": str(exc)}, confidence=0.0)
            raise

        return State.COMPLETED



    def _coerce_to_optimized_plan(self, raw: Any) -> OptimizedPlan:
        if isinstance(raw, OptimizedPlan):
            return raw
        if isinstance(raw, dict):
            return OptimizedPlan.model_validate(raw)
        if hasattr(raw, "model_dump"):
            return OptimizedPlan.model_validate(raw.model_dump())
        if hasattr(raw, "__dict__"):
            return OptimizedPlan.model_validate(vars(raw))
        raise TypeError(
            f"Cannot coerce {type(raw)!r} to OptimizedPlan. "
            "Ensure optimizer/refiner returns a dict or Pydantic model."
        )

    def _coerce_to_signal_graph(self, raw: Any) -> SignalGraph:
        if isinstance(raw, SignalGraph):
            return raw
        if isinstance(raw, dict):
            return SignalGraph.model_validate(raw)
        if hasattr(raw, "model_dump"):
            return SignalGraph.model_validate(raw.model_dump())
        self.logger.log(
            tool="MODELING_WARNING",
            intent=f"Could not coerce {type(raw)!r} to SignalGraph — using empty graph",
            inputs={}, outputs={}, confidence=0.3,
        )
        return SignalGraph(nodes=[], edges=[])

    def _coerce_to_validation_report(self, raw: Any) -> ValidationReport:
        if isinstance(raw, ValidationReport):
            return raw
        if isinstance(raw, dict):
            return ValidationReport.model_validate(raw)
        if hasattr(raw, "model_dump"):
            return ValidationReport.model_validate(raw.model_dump())
        # Fallback: extract passed/violations from duck-typed objects.
        return ValidationReport(
            passed=getattr(raw, "passed", False),
            violations=getattr(raw, "violations", []),
        )

    def _transition(self, next_state: State, trigger: str) -> None:
        from_name = self.prev_state.name if self.prev_state else "INIT"

        event = {
            "from_state": from_name,
            "to_state":   next_state.name,
            "trigger":    trigger,
            "job_id":     self.context["job_id"],
        }

        try:
            self.step_sequencer.record(event)
        except TypeError:
            self.step_sequencer.record(
                from_name,
                next_state.name,
                trigger,
                self.context["job_id"],
            )

        self.logger.log(
            tool="STATE_TRANSITION",
            intent=f"{from_name} -> {next_state.name} via {trigger}",
            inputs=event,
            outputs={"to_state": next_state.name},
            confidence=1.0,
        )

        self.prev_state = self.state

    async def _iter_batches(self, ingestor, limit: int):
        total = 0
        for batch in ingestor.load_data(limit=limit):
            yield batch
            total += len(batch)
            if total >= limit:
                break

    def _to_plan_dict(self) -> dict:

        p = self.optimized_plan
    
        if isinstance(p, dict):
            return p
    
        metadata = p.metadata
    
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()
    
        elif hasattr(metadata, "__dict__"):
            metadata = vars(metadata)
    
        return {
            "actions": p.actions,
            "version": p.version,
            "metadata": metadata,
            "speculative_candidates": p.speculative_candidates,
        }
    
    @staticmethod
    def _generate_fingerprint(df: pd.DataFrame) -> str:
        shape_str = f"{df.shape[0]}:{df.shape[1]}"
        col_str   = ",".join(sorted(df.columns.tolist()))
        dtype_str = ",".join(
            f"{c}:{t}" for c, t in sorted(df.dtypes.astype(str).items())
        )
        raw = f"{shape_str}|{col_str}|{dtype_str}"
        return hashlib.sha256(raw.encode()).hexdigest()