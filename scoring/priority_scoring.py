from typing import List, Type,Dict,Any
from dataclasses import dataclass,field

@dataclass
class ScoringWeights:
    """Configurable weights for different signal attributes"""
    SEVERITY_WEIGHT: Dict[str, float] = field(default_factory=lambda: {
    "critical": 1.00,   
    "high":     0.75,   
    "medium":   0.50,  
    "low":      0.25,  
    "info":     0.10,   
                        })
    
    IMPACT_MAP:Dict[str, float] = field(default_factory=lambda: {
        "block_analysis":        1.00,

        "drop":                  0.90,  
        "safe_to_drop_rows":     0.85,  
        "drop_redundant":        0.85, 

        "exclude_from_modeling": 0.80,  
        "suggest_drop":          0.75,  
        "reclassify_as_id":      0.72,  
        "use_feature_hashing":   0.68,

        "suggest_deduplication": 0.65,  
        "group_into_cluster":    0.62,

        "log_transform":         0.60,  
        "one_hot_encode":        0.58,  
        "group_rare_categories": 0.55, 
        "cast_to_category":      0.52,  
        "cast_to_categorical":   0.52,  
        "label_encode":          0.48,
        
        "route_to_nlp":          0.45, 
        "route_to_nlp_eda":      0.42,  
        "route_to_timeseries_eda": 0.40,
        "route_to_iot_eda":      0.38,
        "route_to_event_eda":    0.35,
        "route_to_image_eda":    0.33,

        "user_decision_required": 0.30, 
        "warning":               0.25,  
        "investigate":           0.20,  
        "check_composite":       0.15,

        "ignore_duplicates":     0.08,  
        "skip_deduplication":    0.05,  
        "generate_plots":        0.05,

    })

    TYPE_WEIGHT: Dict[str, float] = field(default_factory=lambda: {
        "dataset":               1.30,  
        "schema_detection":      1.20,
        "column_health":         1.10,
        "feature_reduction":     1.00,  
        "type_reclassification": 0.90,
        "encoding_strategy":     0.80,
        "preprocessing":         0.70,
        "visualization":         0.50,
        "data_quality":          1.20,  
        "feature_review":        0.80,  
        "model_risk":            1.30,  
        "transformation":        0.70,  
        "routing":               0.60,
    })

    MAGNITUDE_THRESHOLDS = {
        "columns_affected": {
            "high": (10, 0.30),      
            "medium": (5, 0.15),     
            "low": (1, 0.05),       
        },
        "percentage": {
            "critical": (50, 0.40),  
            "high": (20, 0.25),      
        }
    }

# {Final Score} = ({Base} + {Magnitude} +{Context})*{Urgency}*{Decay}
class PriorityScoring:
    def __init__(self, context: Dict[str, Any], weights: ScoringWeights = None):
        self.context = context
        self.weights = weights or ScoringWeights()
    
        self.total_rows = self.context.get("dataset_overview", {}).get("num_rows", 1)
        self.total_cols = self.context.get("dataset_overview", {}).get("num_columns", 1)

    def score_signal(self, signal: Dict[str, Any]) -> float:
        base = self._calculate_base_score(signal)
        magnitude = self._calculate_magnitude_score(signal)
        context_boost = self._calculate_context_boost(signal)
        urgency = self._calculate_urgency_multiplier(signal)
        decay = self._calculate_decay_factor(signal)
        
        final_score = (base + magnitude + context_boost) * urgency * decay
        
        return min(100.0, max(0.0, final_score))

    def rank_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []

        print("before")
        print(signals)
    
        for signal in signals:
            score = self.score_signal(signal)
    
            enriched = {
                **signal,
                "computed_priority": float(score),
                "original_priority": float(signal.get("priority", 0.0)),
            }
    
            scored.append(enriched)
        print("after")
        print(scored)
    
        return sorted(
            scored,
            key=lambda x: (
                -x.get("computed_priority", 0.0),
                -x.get("original_priority", 0.0),
            ),
        )
    

    #Base
    def _calculate_base_score(self, signal: Dict[str, Any]) -> float:
        severity = signal.get("severity", "info")
        action = signal.get("action", "investigate")
        signal_type = signal.get("type", "feature_review")
        severity_weight = self.weights.SEVERITY_WEIGHT.get(severity, 0.10)
        impact_weight = self.weights.IMPACT_MAP.get(action, 0.20)
        type_weight = self.weights.TYPE_WEIGHT.get(signal_type, 0.80)
        base_score = ((severity_weight + impact_weight) / 2) * type_weight * 100
        return min(100, base_score)
    
    def _calculate_magnitude_score(self,signal: Dict[str, Any]) -> float:

        details = signal.get("details", {})
        magnitude_score = 0.0
        affected_cols = self._extract_column_count(details)
        if affected_cols > 0:
            pct_cols = (affected_cols / self.total_cols) * 100
            if pct_cols >= 50:
                magnitude_score += 40  # 0.40 * 100
            elif pct_cols >= 20:
                magnitude_score += 25
            elif pct_cols >= 10:
                magnitude_score += 10
            else:
                magnitude_score += 5

        for key in ["duplicate_percent", "missing_percent", "percent"]:
            if key in details:
                pct = float(details[key])
                if pct >= 50:
                    magnitude_score += 5
                elif pct >= 20:
                    magnitude_score += 3
                elif pct >= 10:
                    magnitude_score += 1
        
        return min(20.0, magnitude_score)

    def _calculate_context_boost(self, signal: Dict[str, Any]) -> float:
        
        boost = 0.0
        signal_type = signal.get("type", "")
        action = signal.get("action", "")
        
        if signal_type == "dataset" and self.total_rows < 1000:
            boost += 3 
            
        if action == "block_analysis":
            boost += 10
            

        if signal_type == "feature_reduction":
            numeric_count = sum(1 for col in self.context.get("columns", []) if col.get("type") == "numeric")
            if numeric_count > 10:
                boost += 5
                
        if action.startswith("route_to_nlp"):
            text_count = sum(1 for col in self.context.get("columns", []) if col.get("type") == "text")
            if (text_count / max(1, self.total_cols)) > 0.5:
                boost += 4
                
    
        return min(10.0, boost)
    
    def _calculate_decay_factor(self, signal: Dict[str, Any]) -> float:
        """Decay score for signals that are less relevant. Range: 0.5 - 1.0."""
        decay = 1.0
        signal_type = signal.get("type", "")
        
        # Visualization signals decay heavily if dataset > 1M rows
        if signal_type == "visualization" and self.total_rows > 1_000_000:
            decay *= 0.6
            
        # Encoding suggestions decay if there are no categorical columns to encode
        if signal_type == "encoding_strategy":
            cat_count = sum(1 for col in self.context.get("columns", []) if col.get("type") == "categorical")
            if cat_count == 0:
                decay *= 0.5
                
        return max(0.5, decay)
    

    def _calculate_urgency_multiplier(self, signal: Dict[str, Any]) -> float:
        action = signal.get("action", "")
        if action == "block_analysis":
            return 1.5  
        
        severity = signal.get("severity", "info")
        urgency_map = {
            "critical": 1.4,
            "high":     1.2,
            "medium":   1.0,
            "low":      0.9,
            "info":     0.8,
        }
        return urgency_map.get(severity, 1.0)
    def _extract_column_count(self, details: Dict[str, Any]) -> int:
        """Extract number of affected columns from the detail metadata."""
        for key in ["affected_columns", "columns", "pairs"]:
            if key in details:
                val = details[key]
                if isinstance(val, list):
                    return len(val)
        return 0


    