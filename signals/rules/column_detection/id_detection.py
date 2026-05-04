from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

# --- R2.1: Definite ID Column ---
class Definite_ID_Column(BaseRule):
    name = "R2.1_Definite_ID_Column"
    priority = 100

    def applies(self, context):
        # Trigger: unique_ratio > 0.99 and possible_id flag exists
        return any(
            col["unique_ratio"] > 0.99 and col["signals"].get("possible_id") 
            for col in context["columns"]
        )

    def run(self, context):
        id_cols = [
            col["name"] for col in context["columns"] 
            if col["unique_ratio"] > 0.99 and col["signals"].get("possible_id")
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="schema_detection",
                action="exclude_from_modeling",
                severity="info",
                message=f"Identifiers detected: {', '.join(id_cols)}. These should be excluded from machine learning models to prevent leakage.",
                details={"affected_columns": id_cols}
                        )
    
class Composite_Key_Candidate(BaseRule):
    name = "R2.2_Composite_Key_Candidate"
    priority = 90 # Slightly lower than Definite ID

    def applies(self, context):
        # Trigger: unique_ratio > 0.95 and cardinality > 1000
        return any(
            col["unique_ratio"] > 0.95 and col["cardinality"] > 1000 
            for col in context["columns"]
        )

    def run(self, context):
        candidates = [
            col["name"] for col in context["columns"] 
            if col["unique_ratio"] > 0.95 and col["cardinality"] > 1000
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="schema_detection",
                action="check_composite",
                severity="low",
                message=(
                    f"High cardinality columns found: {', '.join(candidates)}. "
                    "These may form a composite key when combined with other columns."
                ),
                details={"affected_columns": candidates}
                        )
