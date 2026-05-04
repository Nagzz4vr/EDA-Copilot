from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class LongTextNLP(BaseRule):
    name = "R3.6_Long_Text_NLP"
    priority = 70

    def applies(self, context):
        return any(
            col["type"] == "text" and 
            "long_text" in col["flags"] and 
            col["stats"].get("avg_length", 0) > 100
            for col in context["columns"]
        )

    def run(self, context):
        long_text_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "text" and "long_text" in col["flags"]
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="preprocessing_pipeline",
                action="route_to_nlp",
                severity="medium",
                message=(
                    f"Long unstructured text detected in: {', '.join(long_text_cols)}. "
                    "Routing to NLP pipeline for tokenization, cleaning, and embedding generation."
                ),
                details={"affected_columns": long_text_cols}
                        )
    
class ShortTextAsCategorical(BaseRule):
    name = "R3.7_Short_Text_Categorical"
    priority = 75

    def applies(self, context):
        return any(
            col["type"] == "text" and 
            col["stats"].get("avg_length", 0) < 20 and 
            col["cardinality"] < 100
            for col in context["columns"]
        )

    def run(self, context):
        short_text_cols = [
            col["name"] for col in context["columns"] 
            if col["type"] == "text" and 
            col["stats"].get("avg_length", 0) < 20 and 
            col["cardinality"] < 100
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="type_reclassification",
                action="cast_to_categorical",
                severity="info",
                message=(
                    f"Short text strings detected in: {', '.join(short_text_cols)}. "
                    "These have low cardinality and are best treated as Categorical "
                    "features rather than NLP input."
                ),
                details={"affected_columns": short_text_cols}
                        )