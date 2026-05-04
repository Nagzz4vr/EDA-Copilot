from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class HighMissingRule(BaseRule):
    name = "R1.5_high_missing"
    priority = 100

    def applies(self, context) -> bool:
        return any(col["signals"].get("high_missing") for col in context["columns"])
    def run(self, context):
       
        bad_cols = [
            col["name"] for col in context["columns"] 
            if col["signals"].get("high_missing")
        ]


        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="column_health",
                action="warning",
                severity="high",
                message=f"High missing values (>50%) detected in: {', '.join(bad_cols)}",
                details={"affected_columns": bad_cols}
                        )

class ModerateMissingRule(BaseRule):
    name = "R1.6_moderate_missing"
    priority = 50

    def applies(self, context) -> bool:
    
        return any(
            col["signals"].get("moderate_missing") 
            for col in context["columns"] 
            if col["type"] in ['numeric', 'categorical']
        )

    def run(self, context):
        moderate_cols = [
            col["name"] for col in context["columns"] 
            if col["signals"].get("moderate_missing") and 
            col["type"] in ['numeric', 'categorical']
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="column_health",
                action="warning",
                severity="medium",
                message=f"Moderate missing values (10-50%) detected in: {', '.join(moderate_cols)}",
                details={"affected_columns": moderate_cols}
                        )


class PatternMissingRule(BaseRule):
    name = "R1.7_pattern_missing"
    priority = 10

    def applies(self, context) -> bool:
        return any(
            col["missing_pattern"]["transitions"] < 5 and 
            col["missing_pattern"]["max_consecutive_missing"] > 100
            for col in context["columns"]
        )

    def run(self, context):
        pattern_cols = [
            col["name"] for col in context["columns"] 
            if col["missing_pattern"]["transitions"] < 5 and 
            col["missing_pattern"]["max_consecutive_missing"] > 100
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="column_health",
                action="investigate",
                severity="medium",
                message=f"Non-random missing patterns (blocks of NAs) found in: {', '.join(pattern_cols)}",
                details={"affected_columns": pattern_cols}
                        )
    
class RandomMissingRule(BaseRule):
    name = "R1.8_random_missing"
    priority = 5 

    def applies(self, context) -> bool:

        return any(
            col["missing"]["percent"] < 5 and 
            col["missing_pattern"]["transitions"] > 50
            for col in context["columns"]
        )

    def run(self, context):
       
        safe_to_drop_cols = [
            col["name"] for col in context["columns"] 
            if col["missing"]["percent"] < 5 and 
            col["missing_pattern"]["transitions"] > 50
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="data_cleaning_suggestion",
                action="safe_to_drop_rows",
                severity="low",
                message=(
                    f"Missing values in {', '.join(safe_to_drop_cols)} appear to be Missing Completely "
                    f"at Random (MCAR). It is safe to drop these rows with minimal impact on variance."
                ),
                details={
                    "affected_columns": safe_to_drop_cols,
                    "reasoning": "Missingness is scattered (high transitions) and low volume (< 5%)."
                }
                        )