from datetime import datetime
from observability.trace_logger import TraceLogger

from dataclasses import dataclass
from typing import Dict, List, Optional
from threading import Lock


@dataclass
class ConfidenceEvent:
    step: str
    score: float
    timestamp: datetime


class ConfidenceTracker:
    def __init__(self):
        self.scores: Dict[str, List[ConfidenceEvent]] = {}
        self._lock = Lock()

    def update(self, state_uuid: str, step: str, score: float):
        data = ConfidenceEvent(
            step=step,
            score=score,
            timestamp=datetime.utcnow()
        )

        with self._lock:
            if state_uuid not in self.scores:
                self.scores[state_uuid] = []
            self.scores[state_uuid].append(data)

    def get_trend(self, state_uuid: str) -> List[float]:
        records = self.scores.get(state_uuid, [])
        return [r.score for r in records]

    def get_current_score(self, state_uuid: str) -> Optional[float]:
        records = self.scores.get(state_uuid, [])
        if records:
            return records[-1].score
        return None

    def get_lowest_confidence_step(self, state_uuid: str) -> Optional[dict]:
        records = self.scores.get(state_uuid, [])
        if not records:
            return None

        lowest = min(records, key=lambda x: x.score)
        return {
            "step": lowest.step,
            "score": lowest.score,
            "timestamp": lowest.timestamp
        }

    def get_score_drop(self, state_uuid: str) -> Optional[float]:
        records = self.scores.get(state_uuid, [])
        if len(records) < 2:
            return None

        max_drop = 0.0

        for i in range(1, len(records)):
            drop = records[i - 1].score - records[i].score
            if drop > max_drop:
                max_drop = drop

        return max_drop if max_drop > 0 else None

    def ingest_from_trace_log(self, trace_entry: dict):
        if "confidence" not in trace_entry:
            return

        state_uuid = trace_entry.get("state_uuid") or trace_entry.get("session")
        tool = trace_entry.get("tool")
        confidence = trace_entry.get("confidence")

        if state_uuid and tool and confidence is not None:
            self.update(state_uuid, tool, confidence)