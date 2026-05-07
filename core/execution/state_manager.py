from datetime import datetime
from typing import Optional, Dict, Any


class StateManager:
    def __init__(self,storage,logger=None):
        self.storage = storage
        self.logger = logger

    def save(self,state_uuid: str,dag,result: Dict[str, Any]):
        record = {
            "state_uuid": state_uuid,
            "dag_version": dag.version,
            "nodes_executed": [ n.id for n in dag.nodes],
            "rows_processed": result["rows_processed"],
            "actions_applied": result["actions_applied"],
            "timestamp": (datetime.utcnow().isoformat()),
            "status": "COMPLETED"
        }

        self.storage.write(key=state_uuid,value=record)
        if self.logger:
            self.logger.log(
                tool="STATE_MANAGER",
                intent="Execution state saved",
                inputs={
                    "state_uuid": state_uuid
                },
                outputs={
                    "status": "COMPLETED"
                },
                confidence=1.0
            )

    def load(self,state_uuid: str) -> Optional[dict]:
        return self.storage.read(
            key=state_uuid
        )
    
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