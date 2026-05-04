from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class BinaryCategorical(BaseRule):
    name = "R3.3_Binary_Categorical"
    priority = 60

    def applies(self, context):
        return any(
            col["type"] == "categorical" and col["cardinality"] == 2
            for col in context["columns"]
        )
    
    def run(self, context):
        binary_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "categorical" and col["cardinality"] == 2
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="encoding_strategy",
                action="label_encode",
                severity="info",
                message=f"Binary categories detected in: {', '.join(binary_cols)}. Label encoding (0/1) is recommended to keep the feature space small.",
                details={"affected_columns": binary_cols}
                        )
    
class LowCardOneHot(BaseRule):
    name = "R3.4_Low_Card_One_Hot"
    priority = 55

    def applies(self, context):
        return any(
            col["type"] == "categorical" and 3 <= col["cardinality"] <= 10
            for col in context["columns"]
        )

    def run(self, context):
        ohe_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "categorical" and 3 <= col["cardinality"] <= 10
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="encoding_strategy",
                action="one_hot_encode",
                severity="info",
                message=f"Low-cardinality categories (3-10) found in: {', '.join(ohe_cols)}. One-Hot Encoding is recommended for these features.",
                details={"affected_columns": ohe_cols}
                        )
    
class ImbalancedCategoricalGrouping(BaseRule):
    name = "R3.5_Imbalanced_Categorical"
    priority = 50

    def applies(self, context):

        return any(
            col["type"] == "categorical" and 
            "imbalanced" in col["flags"] and 
            col["stats"].get("imbalanced", False) 
            for col in context["columns"]
        )

    def run(self, context):
        imbalanced_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "categorical" and "imbalanced" in col["flags"]
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="preprocessing_suggestion",
                action="group_rare_categories",
                severity="medium",
                message=(
                    f"Severe imbalance detected in: {', '.join(imbalanced_cols)}. "
                    "One category dominates >90% of the data. Suggest grouping "
                    "all minority categories into a single 'Other' bin."
                ),
                details={"affected_columns": imbalanced_cols, "threshold": "90%"}
                        )