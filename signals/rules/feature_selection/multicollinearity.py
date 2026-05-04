from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class HighCorrelationRedundant(BaseRule):
    name = "R4.1_High_Correlation_Redundant"
    priority = 40

    def applies(self, context):
        return any(
            pair["correlation"] > 0.95 
            for pair in context["top_correlations"]
        )

    def run(self, context):
        redundant_pairs = [
            f"{p['feature_1']} & {p['feature_2']}" 
            for p in context["top_correlations"] if p["correlation"] > 0.95
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="feature_reduction",
                action="drop_redundant",
                severity="high",
                message=(
                    f"Near-perfect correlation (>0.95) detected in: {', '.join(redundant_pairs)}. "
                    "These features are redundant; one from each pair should be removed."
                ),
                details={"pairs": redundant_pairs}
                        )
    
    
class ModerateCorrelationFlag(BaseRule):
    name = "R4.2_Moderate_Correlation"
    priority = 35

    def applies(self, context):
        return any(
            0.75 < pair["correlation"] < 0.95 
            for pair in context["top_correlations"]
        )

    def run(self, context):
        review_pairs = [
            f"{p['feature_1']} & {p['feature_2']} ({p['correlation']})" 
            for p in context["top_correlations"] if 0.75 < p["correlation"] < 0.95
        ]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="feature_review",
                action="user_decision_required",
                severity="medium",
                message=(
                    f"Strong correlation detected in: {', '.join(review_pairs)}. "
                    "Review these pairs; keeping both may cause multicollinearity issues."
                ),
                details={"pairs": review_pairs}
                        )
    
class CorrelationClusterRule(BaseRule):
    name = "R4.3_Correlation_Cluster"
    priority = 30

    def applies(self, context):
        # Count how many times each feature appears in high-correlation pairs
        from collections import Counter
        all_features = []
        for p in context["top_correlations"]:
            if p["correlation"] > 0.75:
                all_features.extend([p["feature_1"], p["feature_2"]])
        
        counts = Counter(all_features)
        return any(count >= 3 for count in counts.values())

    def run(self, context):
        from collections import Counter
        all_features = []
        for p in context["top_correlations"]:
            if p["correlation"] > 0.75:
                all_features.extend([p["feature_1"], p["feature_2"]])
        
        counts = Counter(all_features)
        clusters = [feat for feat, count in counts.items() if count >= 3]

        return RuleOutput(
                rule_name=self.name,
                priority=self.priority,
                type="feature_reduction",
                action="group_into_cluster",
                severity="medium",
                message=(
                    f"Systematic redundancy found! The following features appear in 3+ "
                    f"correlation pairs: {', '.join(clusters)}. Suggest keeping one "
                    "representative and dropping the rest."
                ),
                details={"cluster_centers": clusters}
                        )