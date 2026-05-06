from observability.trace_logger import TraceLogger
from typing import Set
from observability.confidence_tracker import ConfidenceTracker

class AnomalyDetector:
    CONFIDENCE_DROP_THRESHOLD = 0.30
    MIN_SCORE_FLOOR = 0.40
    CONSECUTIVE_LOW_COUNT = 2

    def __init__(self, confidence_tracker: ConfidenceTracker,logger:TraceLogger):
        self.tracker = confidence_tracker
        self.escalated_uuids: Set[str] = set()
        self.logger = logger

    def check(self, state_uuid: str) -> bool:
        """
        Returns True if anomaly detected, False otherwise.
        Called after every confidence update.
        """
        if state_uuid in self.escalated_uuids:
            return False

        trend = self.tracker.get_trend(state_uuid)

        if len(trend) < 2:
            return False

        # Check 1: Single large drop
        drop = trend[-2] - trend[-1]
        if drop >= self.CONFIDENCE_DROP_THRESHOLD:
            self.logger.log(
                tool="ANOMALY_DETECTOR",
                intent="Large confidence drop detected",
                inputs={"state_uuid": state_uuid},
                outputs={"drop": drop, "from": trend[-2], "to": trend[-1]},
                confidence=0.8
            )
            self.escalated_uuids.add(state_uuid)
            return True

        # Check 2: Floor breach
        current = trend[-1]
        if current < self.MIN_SCORE_FLOOR:
            self.logger.log(
                tool="ANOMALY_DETECTOR",
                intent="Confidence below floor",
                inputs={"state_uuid": state_uuid},
                outputs={"current_score": current, "floor": self.MIN_SCORE_FLOOR},
                confidence=0.9
            )
            self.escalated_uuids.add(state_uuid)
            return True

        # Check 3: Consecutive low scores
        if len(trend) >= self.CONSECUTIVE_LOW_COUNT:
            recent = trend[-self.CONSECUTIVE_LOW_COUNT:]
            if all(s < 0.5 for s in recent):
                self.logger.log(
                    tool="ANOMALY_DETECTOR",
                    intent="Sustained low confidence",
                    inputs={"state_uuid": state_uuid},
                    outputs={"recent_scores": recent},
                    confidence=0.85
                )
                self.escalated_uuids.add(state_uuid)
                return True

        return False

    def get_anomaly_reason(self, state_uuid: str) -> str:
        trend = self.tracker.get_trend(state_uuid)

        if not trend:
            return "Unknown anomaly"

        current = trend[-1]

        if current <self.MIN_SCORE_FLOOR:
            return f"Confidence fell to {current:.2f}, below floor of {self.MIN_SCORE_FLOOR}"

        if len(trend) >= 2:
            drop = trend[-2] - trend[-1]
            if drop >= self.CONFIDENCE_DROP_THRESHOLD:
                return f"Confidence dropped {drop:.1%} in one step (from {trend[-2]:.2f} to {current:.2f})"

        return "Sustained low confidence across multiple steps"