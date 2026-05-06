from datetime import datetime
from observability.trace_logger import TraceLogger
import json
from pathlib import Path

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from threading import Lock


@dataclass
class StepEvent:
    step: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class StepSequencer:
    def __init__(self):
        self.timeline: Dict[str, List[StepEvent]] = {}
        self._lock = Lock()

    def record(self, state_uuid: str, step_name: str, metadata: dict = None):
        data = StepEvent(
            step=step_name,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        with self._lock:
            if state_uuid not in self.timeline:
                self.timeline[state_uuid] = []
            self.timeline[state_uuid].append(data)

        try:
            TraceLogger.log(
                tool="STEP_SEQUENCER",
                intent=f"Recorded step {step_name}",
                inputs={"state_uuid": state_uuid},
                outputs={"step": step_name},
                confidence=1.0
            )
        except Exception:
            pass  

    def get_timeline(self, state_uuid: str) -> List[StepEvent]:
        return self.timeline.get(state_uuid, [])

    def get_step_duration(self, state_uuid: str, step_name: str) -> Optional[float]:
        timeline = self.timeline.get(state_uuid, [])
        matching_steps = [s for s in timeline if s.step == step_name]

        if len(matching_steps) < 2:
            return None

        first_ts = matching_steps[0].timestamp
        last_ts = matching_steps[-1].timestamp
        return (last_ts - first_ts).total_seconds()

    def get_slowest_step(self, state_uuid: str) -> Optional[Dict[str, float]]:
        timeline = self.timeline.get(state_uuid, [])
        if len(timeline) < 2:
            return None

        durations: Dict[str, float] = {}

        for i in range(len(timeline) - 1):
            step_name = timeline[i].step
            duration = (
                timeline[i + 1].timestamp - timeline[i].timestamp
            ).total_seconds()

            # keep max duration per step
            durations[step_name] = max(durations.get(step_name, 0), duration)

        slowest = max(durations.items(), key=lambda x: x[1])
        return {"step": slowest[0], "duration_ms": slowest[1]}

    def export_timeline(self, state_uuid: str, output_path: str):
        timeline = self.get_timeline(state_uuid)

        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / f"{state_uuid}_timeline.json", "w") as f:
            json.dump(
                [
                    {
                        "step": e.step,
                        "timestamp": e.timestamp.isoformat(),
                        "metadata": e.metadata,
                    }
                    for e in timeline
                ],
                f,
                indent=2
            )