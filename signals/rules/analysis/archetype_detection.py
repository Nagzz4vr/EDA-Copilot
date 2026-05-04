from typing import Dict, Any
from signals.rules.base_rule import BaseRule,RuleOutput

class TimeSeriesDatasetRule(BaseRule):
    name = "R7.1_Time_Series_Dataset"
    priority = 90

    def applies(self, context) -> bool:
        has_date = any(col["type"] == "datetime" for col in context["columns"])
        is_ordered = context["dataset_health"].get("is_temporally_ordered", False)
        
        return has_date and is_ordered

    def run(self, context):
        date_cols = [col["name"] for col in context["columns"] if col["type"] == "datetime"]

        return RuleOutput(
            rule_name=self.name,
            priority=self.priority,
            type= "workflow_routing",
            action= "route_to_timeseries_eda",
            severity="info",
            details={"temporal_columns": date_cols},
            message=(
                f"Temporal ordering detected using {', '.join(date_cols)}. "
                "Routing to Time-Series workflow to analyze seasonality, "
                "stationarity, and trends."
            )
        )
class TransactionalDatasetRule(BaseRule):
    name = "R6.2_Transactional_Dataset"
    priority = 85

    def applies(self, context) -> bool:
        col_names = [col["name"].lower() for col in context["columns"]]
        has_id = any(name in col_names for name in ["transaction_id", "order_id", "event_id"])
        has_date = any(col["type"] == "datetime" for col in context["columns"])
        
        return has_id and has_date

    def run(self, context):
        return RuleOutput(
            rule_name=self.name,
            priority=self.priority,
            type="workflow_routing",
            action="route_to_event_eda",
            severity="info",
            message=(
                "Transactional schema detected (IDs + Timestamps). "
                "Suggesting event-based analysis: data should be aggregated "
                "by entity (e.g., user_id) or time-window before modeling."
            ),
            details={"logic": "aggregate_by_entity_and_time"}
        )
    
class ImageMetadataRule(BaseRule):
    name = "R6.3_Image_Metadata"
    priority = 85

    def applies(self, context) -> bool:
        col_names = [col["name"].lower() for col in context["columns"]]
        has_img_path = any("image_path" in name or "filename" in name for name in col_names)
        # Check if any text column mentions extensions like .jpg, .png, .svg
        has_ext = any(".jpg" in str(col["stats"].get("top_values", "")) for col in context["columns"])
        return has_img_path or has_ext

    def run(self, context):
        return RuleOutput(
            rule_name=self.name,
            priority=self.priority,
            type="workflow_routing",
            action="route_to_image_eda",
            severity="medium",
            message="Image metadata detected. Workflow adjusted to focus on path extraction and image attribute analysis.",
            details={
                "detected_features": ["file_paths", "image_extensions"],
                "target_module": "ComputerVisionEDA"
            }
                        )   
    
class SensorIoTRule(BaseRule):
    name = "R6.4_Sensor_IoT"
    priority = 80

    def applies(self, context) -> bool:
        num_numeric = sum(1 for col in context["columns"] if col["type"] == "numeric")
        has_date = any(col["type"] == "datetime" for col in context["columns"])
        high_missing = context["dataset_health"].get("duplicate_percent", 0) > 10
        
        return num_numeric > 20 and has_date and high_missing

    def run(self):
        return RuleOutput(
            rule_name=self.name,
            priority=self.priority,
            type="workflow_routing",
            action="route_to_iot_eda",
            severity="high",
            message="IoT/Sensor profile detected. Analysis will focus on time-series interpolation and measurement gap signal analysis.",
            details={
                "numeric_density": "high",
                "temporal_component": True,
                "data_quality_issue": "high_duplicates"
                    }
                    )
    
class NLPDatasetRule(BaseRule):
    name = "R6.5_NLP_Dataset"
    priority = 80

    def applies(self, context) -> bool:
        text_count = sum(1 for col in context["columns"] if col["type"] == "text")
        other_count = sum(1 for col in context["columns"] if col["type"] in ["numeric", "categorical"])
        
        # Calculate average length across all text columns
        avg_lengths = [col["stats"].get("avg_length", 0) for col in context["columns"] if col["type"] == "text"]
        global_avg = sum(avg_lengths) / len(avg_lengths) if avg_lengths else 0
        
        return text_count > other_count and global_avg > 50

    def run(self):
        return RuleOutput(
            rule_name=self.name,
            priority=self.priority,
            type="workflow_routing",
            action="route_to_nlp_eda",
            severity="medium",
            message="Unstructured text dominates this dataset. Routing to NLP workflow for tokenization and corpus analysis.",
            details={
                    "unstructured_ratio": "dominant",
                    "avg_character_length": "long_form_text"
                    }
                        )