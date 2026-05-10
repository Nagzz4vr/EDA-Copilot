from datetime import datetime
from typing import Optional, Dict, Any


class StateManager:
    def __init__(self,storage,logger=None):


        print("STATE MANAGER INIT")
        print("storage =", storage)
        print("storage type =", type(storage))
        self.storage = storage
        self.logger = logger

    def save(self, state_uuid: str, planner_bundle, result: Dict[str, Any]):

        optimized_plan = planner_bundle.optimized_plan
        signal_graph = planner_bundle.signal_graph

        record = {
            "state_uuid": state_uuid,

            "planner_bundle": {
                "optimized_plan": optimized_plan.model_dump()
                if hasattr(optimized_plan, "model_dump")
                else vars(optimized_plan),

                "signal_graph": signal_graph.model_dump()
                if hasattr(signal_graph, "model_dump")
                else vars(signal_graph),

                "validation_report": (
                    planner_bundle.validation_report.model_dump()
                    if getattr(planner_bundle, "validation_report", None) and hasattr(planner_bundle.validation_report, "model_dump")
                    else getattr(planner_bundle, "validation_report", None)
                ),

                "fingerprint": getattr(planner_bundle, "fingerprint", None),
            },

            "actions_executed": [
                getattr(a, "action_type", None)
                for a in optimized_plan.actions
            ] if hasattr(optimized_plan, "actions") else [],

            "rows_processed": result.get("rows_processed"),
            "actions_applied": result.get("actions_applied"),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "COMPLETED",
            "result": result,
        }

        self.storage.write(
            key=state_uuid,
            value=record
        )

        if self.logger:
            self.logger.log(
                tool="STATE_MANAGER",
                intent="Execution state saved",
                inputs={"state_uuid": state_uuid},
                outputs={"status": "COMPLETED"},
                confidence=1.0
            )

    def load(self,state_uuid: str) -> Optional[dict]:
        return self.storage.read(key=state_uuid)
    
    def mark_failed(self,state_uuid: str,error_msg: str):
        record = self.storage.read( key=state_uuid)

        if record is None:
            record = {
                "state_uuid": state_uuid,
                "status": "FAILED"
            }
        record["status"] = "FAILED"
        record["error"] = error_msg
        record["failed_at"] = (datetime.utcnow().isoformat())
        self.storage.write(key=state_uuid,value=record)
        if self.logger:
            self.logger.log(
                tool="STATE_MANAGER",
                intent="Execution marked failed",
                inputs={
                    "state_uuid": state_uuid
                },
                outputs={
                    "error": error_msg
                },
                confidence=0.0
            )