from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class Constant_Column(BaseRule):
    name = "R2.3_Constant_Column"
    priority = 100

    def applies(self, context):
        return any(col["signals"].get("constant") for col in context["columns"])
    
    def run(self, context):
        cnst_cols = [
            col["name"] for col in context["columns"] 
            if col["signals"].get("constant")
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="column_cleaning",
                action="drop",
                severity="high",
                message=f"Constant values detected in: {', '.join(cnst_cols)}. These columns contain zero information and should be dropped.",
                details={"affected_columns": cnst_cols}
                        )
    
class NearConstantNumeric(BaseRule):
    name = "R2.4_Near_Constant_Numeric"
    priority = 90 

    def applies(self, context):
       
        return any(
            col["signals"].get("low_variance") and col["type"] == "numeric" 
            for col in context["columns"]
        )
    
    def run(self, context):
   
        nr_cnst_cols = [
            col["name"] for col in context["columns"] 
            if col["signals"].get("low_variance") and col["type"] == "numeric"
        ]

        return RuleOutput(
            rule_name=self.name,
            priority=self.priority,
            type="column_cleaning",
            action="suggest_drop",
            severity="medium",
            message=f"Near-constant numeric values detected in: {', '.join(nr_cnst_cols)}. Minimal variance suggests low predictive power.",
            details={"affected_columns": nr_cnst_cols}
                        )