from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

# --- R1.1: Critical Duplicate Level ---
class CriticalDuplicateLevel(BaseRule):
    name = "R1.2_critical_duplicates"
    priority = 100  # highest priority (blocking)

    def applies(self, context):
        dup_pct = context["dataset_health"]["duplicate_percent"]
        return dup_pct > 20

    def run(self, context):
        dup_pct = context["dataset_health"]["duplicate_percent"]

        return RuleOutput(
            rule_name=self.name,
            priority=self.priority,
            type="dataset",
            action="block_analysis",
            severity="critical",
            message=f"Duplicate rows are {dup_pct}%. Analysis blocked until deduplication.",
            details={
                "duplicate_percent": dup_pct
            }
                        )


# --- R1.2: High Duplicate Threshold ---
class HighDuplicateThreshold(BaseRule):
    name = "R1.1_high_duplicates"
    priority = 50

    def applies(self, context):
        dup_pct = context["dataset_health"]["duplicate_percent"]
        return 5 < dup_pct <= 20

    def run(self, context):
        dup_pct = context["dataset_health"]["duplicate_percent"]

        return RuleOutput(
                    rule_name=self.name,
                    priority=self.priority,
                    type="dataset",
                    action="suggest_deduplication",
                    severity="medium",
                    message=f"{dup_pct}% duplicate rows detected. Deduplication recommended.",
                    details={
                        "duplicate_percent": dup_pct
                    }
                        )

# --- R1.3: No Duplicates ---
class NoDuplicates(BaseRule):
    name = "R1.3_no_duplicates"
    priority = 10

    def applies(self, context):
        dup_pct = context["dataset_health"]["duplicate_percent"]
        return dup_pct == 0

    def run(self, context):
            return RuleOutput(
                    rule_name=self.name,
                    priority=self.priority,
                    type="dataset",
                    action="skip_deduplication",
                    severity="info",
                    message="No duplicate rows found. Skipping deduplication step.",
                    details={
                        "duplicate_percent": 0
                    }
                            )

class LowDuplicateNoise(BaseRule):
    """
    Handles small duplicate presence (0–5%) to avoid unnecessary actions.
    """
    name = "R1.4_low_duplicate_noise"
    priority = 5

    def applies(self, context):
        dup_pct = context["dataset_health"]["duplicate_percent"]
        return 0 < dup_pct <= 5

    def run(self, context):
        dup_pct = context["dataset_health"]["duplicate_percent"]
        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="dataset",
                action="ignore_duplicates",
                severity="low",
                message=f"{dup_pct}% duplicates detected. Likely negligible.",
                details={
                    "duplicate_percent": dup_pct
                }
                        )