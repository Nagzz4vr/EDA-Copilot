import pytest
from typing import Dict, Any
from signals.rules.base_rule import RuleOutput
from signals.rules.feature_selection.multicollinearity import (
    HighCorrelationRedundant,
    ModerateCorrelationFlag,
    CorrelationClusterRule
)
from signals.rules.analysis.archetype_detection import (
    TimeSeriesDatasetRule,
    TransactionalDatasetRule,
    ImageMetadataRule,
    SensorIoTRule,
    NLPDatasetRule
)


class TestCorrelationRules:
    """Test suite for correlation detection rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {"top_correlations": []}
    
    # HighCorrelationRedundant Tests
    
    def test_high_correlation_applies(self, base_context):
        """Test high correlation rule applies when correlation > 0.95"""
        base_context["top_correlations"] = [
            {"feature_1": "age", "feature_2": "years_old", "correlation": 0.98}
        ]
        
        rule = HighCorrelationRedundant()
        assert rule.applies(base_context) is True
    
    def test_high_correlation_boundary(self, base_context):
        """Test high correlation rule boundary at 0.95"""
        rule = HighCorrelationRedundant()
        
        # At boundary (excluded)
        base_context["top_correlations"] = [
            {"feature_1": "a", "feature_2": "b", "correlation": 0.95}
        ]
        assert rule.applies(base_context) is False
        
        # Just above boundary
        base_context["top_correlations"] = [
            {"feature_1": "a", "feature_2": "b", "correlation": 0.951}
        ]
        assert rule.applies(base_context) is True
    
    def test_high_correlation_does_not_apply(self, base_context):
        """Test high correlation rule doesn't apply when all correlations <= 0.95"""
        base_context["top_correlations"] = [
            {"feature_1": "a", "feature_2": "b", "correlation": 0.85},
            {"feature_1": "c", "feature_2": "d", "correlation": 0.90}
        ]
        
        rule = HighCorrelationRedundant()
        assert rule.applies(base_context) is False
    
    def test_high_correlation_run(self, base_context):
        """Test high correlation rule output"""
        base_context["top_correlations"] = [
            {"feature_1": "age", "feature_2": "years_old", "correlation": 0.99},
            {"feature_1": "price", "feature_2": "cost", "correlation": 0.97}
        ]
        
        rule = HighCorrelationRedundant()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R4.1_High_Correlation_Redundant"
        assert result.priority == 40
        assert result.type == "feature_reduction"
        assert result.action == "drop_redundant"
        assert result.severity == "high"
        assert "age & years_old" in result.message
        assert "price & cost" in result.message
        assert len(result.details["pairs"]) == 2
    
    def test_high_correlation_empty_list(self, base_context):
        """Test high correlation with empty correlation list"""
        base_context["top_correlations"] = []
        
        rule = HighCorrelationRedundant()
        assert rule.applies(base_context) is False
    
    # ModerateCorrelationFlag Tests
    
    def test_moderate_correlation_applies(self, base_context):
        """Test moderate correlation rule applies when 0.75 < correlation < 0.95"""
        base_context["top_correlations"] = [
            {"feature_1": "height", "feature_2": "weight", "correlation": 0.85}
        ]
        
        rule = ModerateCorrelationFlag()
        assert rule.applies(base_context) is True
    
    def test_moderate_correlation_boundaries(self, base_context):
        """Test moderate correlation rule boundaries"""
        rule = ModerateCorrelationFlag()
        
        # Lower boundary (excluded)
        base_context["top_correlations"] = [
            {"feature_1": "a", "feature_2": "b", "correlation": 0.75}
        ]
        assert rule.applies(base_context) is False
        
        # Just above lower boundary
        base_context["top_correlations"] = [
            {"feature_1": "a", "feature_2": "b", "correlation": 0.76}
        ]
        assert rule.applies(base_context) is True
        
        # Just below upper boundary
        base_context["top_correlations"] = [
            {"feature_1": "a", "feature_2": "b", "correlation": 0.94}
        ]
        assert rule.applies(base_context) is True
        
        # Upper boundary (excluded)
        base_context["top_correlations"] = [
            {"feature_1": "a", "feature_2": "b", "correlation": 0.95}
        ]
        assert rule.applies(base_context) is False
    
    def test_moderate_correlation_run(self, base_context):
        """Test moderate correlation rule output"""
        base_context["top_correlations"] = [
            {"feature_1": "income", "feature_2": "spending", "correlation": 0.82}
        ]
        
        rule = ModerateCorrelationFlag()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R4.2_Moderate_Correlation"
        assert result.priority == 35
        assert result.type == "feature_review"
        assert result.action == "user_decision_required"
        assert result.severity == "medium"
        assert "income & spending" in result.message
        assert "0.82" in result.message
    
    # CorrelationClusterRule Tests
    
    def test_correlation_cluster_applies(self, base_context):
        """Test correlation cluster rule applies when feature appears 3+ times"""
        base_context["top_correlations"] = [
            {"feature_1": "feat_a", "feature_2": "feat_b", "correlation": 0.80},
            {"feature_1": "feat_a", "feature_2": "feat_c", "correlation": 0.85},
            {"feature_1": "feat_a", "feature_2": "feat_d", "correlation": 0.78}
        ]
        
        rule = CorrelationClusterRule()
        assert rule.applies(base_context) is True
    
    def test_correlation_cluster_does_not_apply(self, base_context):
        """Test correlation cluster rule doesn't apply with feature appearing < 3 times"""
        base_context["top_correlations"] = [
            {"feature_1": "feat_a", "feature_2": "feat_b", "correlation": 0.80},
            {"feature_1": "feat_a", "feature_2": "feat_c", "correlation": 0.85}
        ]
        
        rule = CorrelationClusterRule()
        assert rule.applies(base_context) is False
    
    def test_correlation_cluster_counts_both_positions(self, base_context):
        """Test correlation cluster counts features in both positions"""
        base_context["top_correlations"] = [
            {"feature_1": "feat_a", "feature_2": "feat_b", "correlation": 0.80},
            {"feature_1": "feat_c", "feature_2": "feat_a", "correlation": 0.85},
            {"feature_1": "feat_d", "feature_2": "feat_a", "correlation": 0.78}
        ]
        
        rule = CorrelationClusterRule()
        assert rule.applies(base_context) is True
    
    def test_correlation_cluster_run(self, base_context):
        """Test correlation cluster rule output"""
        base_context["top_correlations"] = [
            {"feature_1": "price_usd", "feature_2": "price_eur", "correlation": 0.90},
            {"feature_1": "price_usd", "feature_2": "price_gbp", "correlation": 0.88},
            {"feature_1": "price_usd", "feature_2": "price_jpy", "correlation": 0.92}
        ]
        
        rule = CorrelationClusterRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R4.3_Correlation_Cluster"
        assert result.priority == 30
        assert result.type == "feature_reduction"
        assert result.action == "group_into_cluster"
        assert result.severity == "medium"
        assert "price_usd" in result.message
        assert "price_usd" in result.details["cluster_centers"]
    
    def test_correlation_cluster_multiple_clusters(self, base_context):
        """Test correlation cluster with multiple cluster centers"""
        base_context["top_correlations"] = [
            # Cluster 1: price_usd
            {"feature_1": "price_usd", "feature_2": "price_eur", "correlation": 0.90},
            {"feature_1": "price_usd", "feature_2": "price_gbp", "correlation": 0.88},
            {"feature_1": "price_usd", "feature_2": "price_jpy", "correlation": 0.92},
            # Cluster 2: volume
            {"feature_1": "volume_ml", "feature_2": "volume_oz", "correlation": 0.85},
            {"feature_1": "volume_ml", "feature_2": "volume_l", "correlation": 0.87},
            {"feature_1": "volume_ml", "feature_2": "volume_gal", "correlation": 0.83}
        ]
        
        rule = CorrelationClusterRule()
        result = rule.run(base_context)
        
        assert len(result.details["cluster_centers"]) == 2
        assert "price_usd" in result.details["cluster_centers"]
        assert "volume_ml" in result.details["cluster_centers"]
    
    # Integration Tests
    
    def test_correlation_rule_priorities(self):
        """Test that correlation rules have correct priority ordering"""
        high = HighCorrelationRedundant()
        moderate = ModerateCorrelationFlag()
        cluster = CorrelationClusterRule()
        
        assert high.priority > moderate.priority > cluster.priority


class TestWorkflowRoutingRules:
    """Test suite for workflow routing rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {
            "columns": [],
            "dataset_health": {}
        }
    
    @pytest.fixture
    def column_template(self):
        """Template for column structure"""
        return {
            "name": "test_col",
            "type": "numeric",
            "stats": {}
        }
    
    # TimeSeriesDatasetRule Tests
    
    def test_timeseries_applies(self, base_context, column_template):
        """Test time series rule applies with datetime column and temporal ordering"""
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        datetime_col["name"] = "timestamp"
        
        base_context["columns"] = [datetime_col]
        base_context["dataset_health"]["is_temporally_ordered"] = True
        
        rule = TimeSeriesDatasetRule()
        assert rule.applies(base_context) is True
    
    def test_timeseries_requires_datetime(self, base_context, column_template):
        """Test time series rule requires datetime column"""
        base_context["columns"] = [column_template]
        base_context["dataset_health"]["is_temporally_ordered"] = True
        
        rule = TimeSeriesDatasetRule()
        assert rule.applies(base_context) is False
    
    def test_timeseries_requires_temporal_ordering(self, base_context, column_template):
        """Test time series rule requires temporal ordering"""
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = [datetime_col]
        base_context["dataset_health"]["is_temporally_ordered"] = False
        
        rule = TimeSeriesDatasetRule()
        assert rule.applies(base_context) is False
    
    def test_timeseries_run(self, base_context, column_template):
        """Test time series rule output"""
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        datetime_col["name"] = "date"
        
        base_context["columns"] = [datetime_col]
        base_context["dataset_health"]["is_temporally_ordered"] = True
        
        rule = TimeSeriesDatasetRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R7.1_Time_Series_Dataset"
        assert result.priority == 90
        assert result.type == "workflow_routing"
        assert result.action == "route_to_timeseries_eda"
        assert result.severity == "info"
        assert "date" in result.message
        assert result.details["temporal_columns"] == ["date"]
    
    # TransactionalDatasetRule Tests
    
    def test_transactional_applies(self, base_context, column_template):
        """Test transactional rule applies with transaction ID and datetime"""
        id_col = column_template.copy()
        id_col["name"] = "transaction_id"
        
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = [id_col, datetime_col]
        
        rule = TransactionalDatasetRule()
        assert rule.applies(base_context) is True
    
    def test_transactional_order_id(self, base_context, column_template):
        """Test transactional rule recognizes order_id"""
        id_col = column_template.copy()
        id_col["name"] = "order_id"
        
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = [id_col, datetime_col]
        
        rule = TransactionalDatasetRule()
        assert rule.applies(base_context) is True
    
    def test_transactional_event_id(self, base_context, column_template):
        """Test transactional rule recognizes event_id"""
        id_col = column_template.copy()
        id_col["name"] = "EVENT_ID"  # Test case insensitivity
        
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = [id_col, datetime_col]
        
        rule = TransactionalDatasetRule()
        assert rule.applies(base_context) is True
    
    def test_transactional_requires_both(self, base_context, column_template):
        """Test transactional rule requires both ID and datetime"""
        rule = TransactionalDatasetRule()
        
        # Only ID, no datetime
        id_col = column_template.copy()
        id_col["name"] = "transaction_id"
        base_context["columns"] = [id_col]
        assert rule.applies(base_context) is False
        
        # Only datetime, no ID
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        base_context["columns"] = [datetime_col]
        assert rule.applies(base_context) is False
    
    def test_transactional_run(self, base_context, column_template):
        """Test transactional rule output"""
        id_col = column_template.copy()
        id_col["name"] = "order_id"
        
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = [id_col, datetime_col]
        
        rule = TransactionalDatasetRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R6.2_Transactional_Dataset"
        assert result.priority == 85
        assert result.action == "route_to_event_eda"
        assert result.details["logic"] == "aggregate_by_entity_and_time"
    
    # ImageMetadataRule Tests
    
    def test_image_metadata_applies_image_path(self, base_context, column_template):
        """Test image metadata rule applies with image_path column"""
        img_col = column_template.copy()
        img_col["name"] = "image_path"
        img_col["stats"]["top_values"] = []
        
        base_context["columns"] = [img_col]
        
        rule = ImageMetadataRule()
        assert rule.applies(base_context) is True
    
    def test_image_metadata_applies_filename(self, base_context, column_template):
        """Test image metadata rule applies with filename column"""
        img_col = column_template.copy()
        img_col["name"] = "photo_filename"
        img_col["stats"]["top_values"] = []
        
        base_context["columns"] = [img_col]
        
        rule = ImageMetadataRule()
        assert rule.applies(base_context) is True
    
    def test_image_metadata_applies_jpg_extension(self, base_context, column_template):
        """Test image metadata rule applies with .jpg extension in data"""
        img_col = column_template.copy()
        img_col["name"] = "file"
        img_col["stats"]["top_values"] = "photo1.jpg"
        
        base_context["columns"] = [img_col]
        
        rule = ImageMetadataRule()
        assert rule.applies(base_context) is True
    
    def test_image_metadata_run(self, base_context, column_template):
        """Test image metadata rule output"""
        img_col = column_template.copy()
        img_col["name"] = "image_path"
        img_col["stats"]["top_values"] = []
        
        base_context["columns"] = [img_col]
        
        rule = ImageMetadataRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R6.3_Image_Metadata"
        assert result.priority == 85
        assert result.action == "route_to_image_eda"
        assert result.details["target_module"] == "ComputerVisionEDA"
    
    # SensorIoTRule Tests
    
    def test_sensor_iot_applies(self, base_context, column_template):
        """Test sensor IoT rule applies with many numeric columns, datetime, and high duplicates"""
        # Create 25 numeric columns
        numeric_cols = []
        for i in range(25):
            col = column_template.copy()
            col["name"] = f"sensor_{i}"
            col["type"] = "numeric"
            numeric_cols.append(col)
        
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = numeric_cols + [datetime_col]
        base_context["dataset_health"]["duplicate_percent"] = 15
        
        rule = SensorIoTRule()
        assert rule.applies(base_context) is True
    
    def test_sensor_iot_requires_enough_numeric(self, base_context, column_template):
        """Test sensor IoT rule requires > 20 numeric columns"""
        rule = SensorIoTRule()
        
        # Create 20 numeric columns (boundary, excluded)
        numeric_cols = []
        for i in range(20):
            col = column_template.copy()
            col["name"] = f"sensor_{i}"
            col["type"] = "numeric"
            numeric_cols.append(col)
        
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = numeric_cols + [datetime_col]
        base_context["dataset_health"]["duplicate_percent"] = 15
        
        assert rule.applies(base_context) is False
        
        # Add one more (21 total)
        col = column_template.copy()
        col["name"] = "sensor_21"
        col["type"] = "numeric"
        base_context["columns"].append(col)
        
        assert rule.applies(base_context) is True
    
    def test_sensor_iot_run(self, base_context, column_template):
        """Test sensor IoT rule output"""
        # Create 25 numeric columns
        numeric_cols = []
        for i in range(25):
            col = column_template.copy()
            col["name"] = f"sensor_{i}"
            col["type"] = "numeric"
            numeric_cols.append(col)
        
        datetime_col = column_template.copy()
        datetime_col["type"] = "datetime"
        
        base_context["columns"] = numeric_cols + [datetime_col]
        base_context["dataset_health"]["duplicate_percent"] = 15
        
        rule = SensorIoTRule()
        result = rule.run()
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R6.4_Sensor_IoT"
        assert result.priority == 80
        assert result.action == "route_to_iot_eda"
        assert result.details["numeric_density"] == "high"
    
    # NLPDatasetRule Tests
    
    def test_nlp_dataset_applies(self, base_context, column_template):
        """Test NLP dataset rule applies when text columns dominate"""
        text_col1 = column_template.copy()
        text_col1["name"] = "review"
        text_col1["type"] = "text"
        text_col1["stats"]["avg_length"] = 150
        
        text_col2 = column_template.copy()
        text_col2["name"] = "comment"
        text_col2["type"] = "text"
        text_col2["stats"]["avg_length"] = 200
        
        numeric_col = column_template.copy()
        numeric_col["name"] = "rating"
        numeric_col["type"] = "numeric"
        
        base_context["columns"] = [text_col1, text_col2, numeric_col]
        
        rule = NLPDatasetRule()
        assert rule.applies(base_context) is True
    
    def test_nlp_dataset_requires_long_text(self, base_context, column_template):
        """Test NLP dataset rule requires avg_length > 50"""
        rule = NLPDatasetRule()
        
        text_col1 = column_template.copy()
        text_col1["name"] = "text1"
        text_col1["type"] = "text"
        text_col1["stats"]["avg_length"] = 40
        
        text_col2 = column_template.copy()
        text_col2["name"] = "text2"
        text_col2["type"] = "text"
        text_col2["stats"]["avg_length"] = 40
        
        base_context["columns"] = [text_col1, text_col2]
        
        assert rule.applies(base_context) is False
    
    def test_nlp_dataset_run(self, base_context, column_template):
        """Test NLP dataset rule output"""
        text_col = column_template.copy()
        text_col["name"] = "description"
        text_col["type"] = "text"
        text_col["stats"]["avg_length"] = 200
        
        base_context["columns"] = [text_col]
        
        rule = NLPDatasetRule()
        result = rule.run()
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R6.5_NLP_Dataset"
        assert result.priority == 80
        assert result.action == "route_to_nlp_eda"
        assert result.details["unstructured_ratio"] == "dominant"

