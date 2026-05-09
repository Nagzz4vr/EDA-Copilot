from enum import Enum, auto
from typing import Dict,Optional
import pandas as pd
import hashlib
class State(Enum):
    PERCEIVING = auto()
    SIGNALING = auto()
    MODELING = auto()
    OPTIMIZING = auto()
    SIMULATING = auto()
    REFINING = auto()
    VALIDATING = auto()
    EXECUTING = auto()
    FINALIZING = auto()
    COMPLETED = auto()
    FAILED = auto()

class MetaAgent:
    def __init__(self, job_context, **kwargs):
        self.context = job_context
        self.state = State.PERCEIVING
        self.prev_state: Optional[State] = None   
        self.max_iterations = 3
        self.iteration_count = 0

        self.ingestor = kwargs["ingestor"]
        self.canonicalizer = kwargs["canonicalizer"]
        self.cache_manager = kwargs["cache_manager"]
        self.rule_engine = kwargs["rule_engine"]
        self.graph_builder = kwargs["graph_builder"]
        self.optimizer = kwargs["optimizer"]
        self.simulator = kwargs["simulator"]
        self.refiner = kwargs["refiner"]
        self.validator = kwargs["validator"]
        self.executor = kwargs["executor"]
        self.logger = kwargs["logger"]
        self.context_builder = kwargs["context_builder"]
        self.variable_selector = kwargs["variable_selector"]
 
        self.plan = None
        self.signal_bag = None
        self.signal_graph = None
        self.optimize_plan = None
        self.canonical_output = None
        self.state_uuid: Optional[str] = None 

    async def run(self):
        """The main state machine loop."""
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
 
                elif self.state == State.OPTIMIZING:
                    self._transition(State.OPTIMIZING, trigger="GRAPH_COMPILED")
                    self.state = await self._optimize()
 
                elif self.state == State.SIMULATING:
                    self._transition(State.SIMULATING, trigger="PLAN_OPTIMIZED")   
                    self.state = await self._simulate()
 
                elif self.state == State.REFINING:
                    self._transition(State.REFINING, trigger="PLAN_VIOLATION_DETECTED")
                    self.state = await self._refine()
 
                elif self.state == State.VALIDATING:
                    self._transition(State.VALIDATING, trigger="CONVERGENCE_REACHED")
                    self.state = await self._validate()
 
                elif self.state == State.EXECUTING:
                    self._transition(State.EXECUTING, trigger="VALIDATION_PASSED")
                    self.state = await self._execute()
 
                elif self.state == State.FINALIZING:
                    self._transition(State.FINALIZING, trigger="EXECUTION_PASSED")
                    self.state = await self._finalize()

        except Exception as e:
            prev = self.state
            self.state = State.FAILED
            self._transition(State.FAILED, trigger="CONTRACT_VIOLATION")
            print(f"FATAL ERROR in state {prev.name}: {e}")
 
        print(f"Job {self.context['job_id']} finished with state: {self.state.name}")

    async def _perceive(self) -> State:
        self.logger.log(tool="INGESTOR",intent="Starting perception phase",inputs={},outputs={},confidence=1.0)
        try:
            selector = self.variable_selector(self.context["file_path"])
            columns = selector.load_column_names()
            target_col = self.context.get("target_column") or columns[-1]
 
            ingestor = self.ingestor(self.context["file_path"])
            batches = [batch for batch in ingestor.load_data(limit=50_000)]
            full_df = pd.concat(batches, ignore_index=True)
 
            builder = self.context_builder(full_df, target_col=target_col)
            raw_analysis_dict = builder.build_context()
 
            fingerprint = self._generate_fingerprint(full_df)  
            self.state_uuid = fingerprint                       
 
            complete_payload = {
                "metadata": {
                    "state_uuid": self.state_uuid,
                    "fingerprint": fingerprint,
                    "schema_version": "1.0.0",
                },
                "canonical_data": raw_analysis_dict,
            }
 
            self.canonical_output = self.validator.validate_canonical(complete_payload)
 
            cache_hit = await self.cache_manager.check(
                self.canonical_output.metadata.fingerprint
            )
            if cache_hit:
                self.plan = cache_hit.plan
                return State.FINALIZING
 
            return State.SIGNALING
 
        except Exception as e:
            self.logger.log(tool="PERCEPTION_ERROR",intent="Perception phase failed",inputs={},outputs={"error": str(e)},confidence=0.0)
            raise


    async def _signal(self) -> State:
        self.logger.log(tool="RULE_ENGINE",intent="Starting signaling phase",inputs={},outputs={},confidence=1.0)
        try:
            engine = self.rule_engine(context=self.canonical_output.model_dump())
            self.signal_bag = engine.run()
 
            if not self.signal_bag:
                self.logger.log(
                    tool="RULE_ENGINE",
                    intent="No signals generated from rules",
                    inputs={},
                    outputs={},
                    confidence=1.0,
                )
        except Exception as e:
            self.logger.log(
                tool="SIGNALING_ERROR",
                intent="Signaling phase failed",
                inputs={},
                outputs={"error": str(e)},
                confidence=0.0,
            )
            raise
 
        return State.MODELING
    

    async def _model(self) -> State:
        self.logger.log(tool="SIGNAL_GRAPH_BUILDER",intent="Starting graph building",inputs={},outputs={},confidence=1.)
        try:
            self.signal_graph = self.graph_builder._build_graph(self.signal_bag)
 
            if not self.signal_graph:
                self.logger.log(tool="GRAPH_BUILDER",intent="No graph generated",inputs={},outputs={},confidence=1.0)
        except Exception as e:
            self.logger.log(tool="MODELING_ERROR",intent="Modeling phase failed",inputs={},outputs={"error": str(e)},confidence=0.0)
            raise
        return State.OPTIMIZING



    async def _optimize(self) -> State:
        self.logger.log(tool="PLAN_OPTIMIZER",intent="Starting optimizing",inputs={},outputs={},confidence=1.0 )
        try:
            self.optimize_plan = self.optimizer.optimize(self.signal_graph)
        except Exception as e:
            self.logger.log(tool="OPTIMIZING_ERROR",intent="Optimization phase failed",inputs={},outputs={"error": str(e)},confidence=0.0)
            raise
 
        return State.SIMULATING


    async def _simulate(self) -> State:
        self.logger.log(tool="SIMULATOR",intent="Starting simulation",inputs={},outputs={},confidence=1.0)
        try:
            self.impact_report = self.simulator.simulate()    
        except Exception as e:
            self.logger.log(tool="SIMULATING_ERROR",intent="Simulation phase failed",inputs={},outputs={"error": str(e)},confidence=0.0)
            raise

        return State.REFINING

    async def _refine(self) -> State:
        self.logger.log(tool="PLAN_REFINER",intent="Starting refinement",inputs={},outputs={},confidence=1.0)
        try:
            self.plan = self.refiner.refine()                 
            self.iteration_count += 1
 
            if self.iteration_count >= self.max_iterations:
                return State.VALIDATING
 
        except Exception as e:
            self.logger.log(
                tool="REFINING_ERROR",
                intent="Refinement phase failed",
                inputs={},
                outputs={"error": str(e)},
                confidence=0.0,
            )
            raise                                             
        return State.SIMULATING

    async def _validate(self) -> State:
        self.logger.log(tool="VALIDATOR",intent="Starting validation",inputs={"plan_version": self.plan.get("version") if self.plan else None},outputs={},confidence=1.0)
        try:
            validation_report = self.validator.validate_plan(self.plan)

            if validation_report.passed:
                self.logger.log(tool="VALIDATOR",intent="Validation passed",inputs={},outputs={"status": "PASSED"},confidence=1.0)
                return State.EXECUTING
            has_critical = any(
                v.get("severity") == "CRITICAL"
                for v in validation_report.violations
            )
            iterations_remaining = self.iteration_count < self.max_iterations

            if not has_critical and iterations_remaining:
                self.logger.log(tool="VALIDATOR",intent="Validation failed — routing back to refinement",inputs={},
                    outputs={
                        "violations": validation_report.violations,
                        "iterations_remaining": self.max_iterations - self.iteration_count,
                    },
                    confidence=1.0,
                )
                return State.REFINING

            self.logger.log(tool="VALIDATOR",intent="Validation failed — unrecoverable",inputs={},
                            outputs={ "violations": validation_report.violations,   
                                      "has_critical": has_critical,  
                                          "iterations_remaining": self.max_iterations - self.iteration_count,
                                },
                confidence=0.0,
            )
            raise RuntimeError(
                f"Validation failed with unrecoverable violations: {validation_report.violations}"
            )

        except RuntimeError:
            raise
        except Exception as e:
            self.logger.log(tool="VALIDATION_ERROR",intent="Validation phase raised an unexpected exception",inputs={},outputs={"error": str(e)},confidence=0.0)
            raise

    async def _execute(self) -> State:
        self.logger.log(tool="DAG_EXECUTOR",intent="Starting execution",inputs={"plan_version": self.plan.get("version") if self.plan else None},outputs={},confidence=1.0)
        try:
            self.execution_result = await self.executor.run(plan=self.plan,mode="FINAL",state_uuid=self.state_uuid,)

            self.logger.log(tool="DAG_EXECUTOR",intent="Execution completed",inputs={},
                outputs={
                    "rows_processed": self.execution_result.get("rows_processed"),
                    "actions_applied": self.execution_result.get("actions_applied"),
                },
                confidence=1.0,
            )

        except Exception as e:
            self.logger.log(tool="EXECUTION_ERROR",intent="Execution phase failed",inputs={},outputs={"error": str(e)},confidence=0.0)
            raise

        return State.FINALIZING

    async def _finalize(self) -> State:
        self.logger.log(tool="FINALIZER",intent="Starting finalization",inputs={},outputs={},confidence=1.0)
        try:
            await self.executor.write_outputs(result=self.execution_result,job_id=self.context["job_id"])
            await self.cache_manager.store(fingerprint=self.canonical_output.metadata.fingerprint,plan=self.plan,execution_result=self.execution_result)
            self.logger.log(tool="FINALIZER",intent="Job completed successfully",
                inputs={
                    "job_id": self.context["job_id"],
                    "state_uuid": self.state_uuid,
                },
                outputs={
                    "plan_version": self.plan.get("version") if self.plan else None,
                    "rows_processed": self.execution_result.get("rows_processed"),
                    "actions_applied": self.execution_result.get("actions_applied"),
                },
                confidence=1.0,
            )

        except Exception as e:
            self.logger.log(tool="FINALIZATION_ERROR",intent="Finalization phase failed",inputs={},outputs={"error": str(e)},confidence=0.0)
            raise

        return State.COMPLETED

    def _transition(self, next_state: State, trigger: str):
        from_name = self.prev_state.name if self.prev_state else "INIT"
        self.logger.log(
            tool="STATE_TRANSITION",
            intent=f"Entering {next_state.name} from {from_name}",
            inputs={
                "from_state": from_name,
                "trigger": trigger,
                "plan_version": self.plan.get("version") if self.plan else None, 
                "state_uuid": self.state_uuid,
            },
            outputs={
                "to_state": next_state.name,
            },
            confidence=1.0,
        )
        self.prev_state = next_state                            
 
    def _generate_fingerprint(self, df: pd.DataFrame) -> str:  
        shape_str = f"{df.shape[0]}:{df.shape[1]}"
        col_str = ",".join(sorted(df.columns.tolist()))
        dtype_str = ",".join(f"{c}:{t}" for c, t in sorted(df.dtypes.astype(str).items()))
        raw = f"{shape_str}|{col_str}|{dtype_str}"
        return hashlib.sha256(raw.encode()).hexdigest()