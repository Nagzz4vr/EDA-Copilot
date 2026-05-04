from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class NumericIDMasquerade(BaseRule):
    name = "R2.5_Numeric_ID_Masquerade"
    priority = 85

    def applies(self, context):
        return any(
            col["type"] == "numeric" and col["unique_ratio"] > 0.95 
            for col in context["columns"]
        )

    def run(self, context):
        id_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "numeric" and col["unique_ratio"] > 0.95
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="type_reclassification",
                action="reclassify_as_id",
                severity="medium",
                message=f"Numeric columns {', '.join(id_cols)} have very high uniqueness. Re-classify as Categorical IDs to prevent the model from treating them as continuous scales.",
                details={"affected_columns": id_cols}
                        )
    
class LowCardNumericToCategorical(BaseRule):
    name = "R2.6_Low_Card_Numeric_Categorical"
    priority = 80

    def applies(self, context):
        return any(
            col["type"] == "numeric" and 
            col["cardinality"] < 10 and 
            "event_like" in col["flags"]
            for col in context["columns"]
        )

    def run(self, context):
        cat_numeric_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "numeric" and 
            col["cardinality"] < 10 and 
            "event_like" in col["flags"]
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="type_reclassification",
                action="cast_to_category",
                severity="info",
                message=f"Columns {', '.join(cat_numeric_cols)} contain few unique integers. These appear to be coded categories and should be treated as Categorical.",
                details={"affected_columns": cat_numeric_cols}
                        )
    
class HighCardCategoricalToText(BaseRule):
    name = "R2.7_High_Card_Categorical_Text"
    priority = 75

    def applies(self, context):
        return any(
            col["type"] == "categorical" and 
            "high_cardinality" in col["flags"] and 
            col["cardinality"] > 1000
            for col in context["columns"]
        )

    def run(self, context):
        high_card_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "categorical" and 
            "high_cardinality" in col["flags"] and 
            col["cardinality"] > 1000
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="encoding_warning",
                action="use_feature_hashing",
                severity="medium",
                message=(
                    f"High cardinality detected in {', '.join(high_card_cols)}. "
                    "One-Hot Encoding these will create too many features. Consider "
                    "Target Encoding, Feature Hashing, or Text Embeddings."
                ),
                details={"affected_columns": high_card_cols}
                        )