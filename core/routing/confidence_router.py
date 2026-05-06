from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class ConfidenceTier(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceRouter:
    THRESHOLDS = {
        "high": 0.85,
        "medium": 0.60,
        "low": 0.0
    }

    SEVERITY_SCORES = {
        "LOW": 0.2,
        "MEDIUM": 0.5,
        "HIGH": 0.8,
        "CRITICAL": 1.0
    }

    @staticmethod
    def classify(blob) -> ConfidenceTier:
        score = ConfidenceRouter._compute_score(blob)

        if score >= ConfidenceRouter.THRESHOLDS["high"]:
            return ConfidenceTier.HIGH
        elif score >= ConfidenceRouter.THRESHOLDS["medium"]:
            return ConfidenceTier.MEDIUM
        else:
            return ConfidenceTier.LOW

    @staticmethod
    def _compute_score(blob) -> float:
        signals = blob.signals

        # --- Extract features from signals ---
        signal_count = len(signals)

        max_severity = 0.0
        leakage_flag = False
        missing_ratios = []

        for s in signals:
            sev = s.get("severity", "LOW")
            max_severity = max(
                max_severity,
                ConfidenceRouter.SEVERITY_SCORES.get(sev, 0.5)
            )

            if s.get("leakage", False):
                leakage_flag = True

            if "missing_ratio" in s:
                missing_ratios.append(s["missing_ratio"])

        # --- Hard rule ---
        if leakage_flag:
            return 0.1  # force LOW confidence

        # --- Normalize components ---
        max_signals_ref = 20

        count_penalty = min(signal_count / max_signals_ref, 1.0)
        severity_penalty = max_severity
        missing_penalty = (
            sum(missing_ratios) / len(missing_ratios)
            if missing_ratios else 0.0
        )

        # --- Weights ---
        w_count = 0.25
        w_severity = 0.35
        w_missing = 0.40

        # --- Aggregate risk ---
        risk = (
            w_count * count_penalty +
            w_severity * severity_penalty +
            w_missing * missing_penalty
        )
        confidence = 1.0 - risk

        # Nonlinear suppression for high missing ratios
        confidence *= (1 - missing_penalty) ** 2

        return max(0.0, min(confidence, 1.0))