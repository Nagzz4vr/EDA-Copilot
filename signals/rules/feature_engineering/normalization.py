from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class HighlySkewedNumeric(BaseRule):
    name = "R3.1_Highly_Skewed_Numeric"
    priority = 85

    def applies(self, context: Dict[str, Any]) -> bool:
        return any(
            col["type"] == "numeric" 
            and col["signals"].get("skewed") 
            and col["stats"]["skew"] > 1.5 
            and col["stats"]["min"] > 0
            for col in context["columns"]
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        skewed_cols = [
            col["name"] for col in context["columns"]
            if col["type"] == "numeric" 
            and col["stats"]["skew"] > 1.5 
            and col["stats"]["min"] > 0
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="normalization_recommendation",
                action="log_transform",
                severity="medium",
                message=(
                    f"Columns {', '.join(skewed_cols)} are highly right-skewed. "
                    "Applying a log transformation is recommended to normalize the distribution "
                    "and improve model convergence."
                ),
                details={
                    "affected_columns": skewed_cols,
                    "transformation": "np.log1p"
                }
                        )