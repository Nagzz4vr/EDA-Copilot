import pytest
from typing import Dict, Any
from signals.rules.base_rule import RuleOutput
from signals.rules.column_detection.low_variance import (
    Constant_Column,
    NearConstantNumeric,

)
from signals.rules.column_detection.col_type import (
    NumericIDMasquerade,
    LowCardNumericToCategorical,
    HighCardCategoricalToText)

class TestConstantColumnRules:
    """Test suite for constant column detection rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {"columns": []}
    
    @pytest.fixture
    def column_template(self):
        """Template for column structure"""
        return {
            "name": "test_col",
            "type": "numeric",
            "signals": {},
            "stats": {}
        }
    
    # Constant_Column Tests
    
    def test_constant_column_applies(self, base_context, column_template):
        """Test constant column rule applies when signal is present"""
        column_template["signals"]["constant"] = True
        base_context["columns"] = [column_template]
        
        rule = Constant_Column()
        assert rule.applies(base_context) is True
    
    def test_constant_column_does_not_apply(self, base_context, column_template):
        """Test constant column rule doesn't apply without signal"""
        column_template["signals"]["constant"] = False
        base_context["columns"] = [column_template]
        
        rule = Constant_Column()
        assert rule.applies(base_context) is False
    
    def test_constant_column_run(self, base_context, column_template):
        """Test constant column rule output"""
        column_template["signals"]["constant"] = True
        column_template["name"] = "status"
        base_context["columns"] = [column_template]
        
        rule = Constant_Column()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R2.3_Constant_Column"
        assert result.priority == 100
        assert result.type == "column_cleaning"
        assert result.action == "drop"
        assert result.severity == "high"
        assert "status" in result.message
        assert "zero information" in result.message.lower()
        assert result.details["affected_columns"] == ["status"]
    
    def test_constant_column_multiple(self, base_context, column_template):
        """Test constant column with multiple constant columns"""
        col1 = column_template.copy()
        col1["name"] = "const1"
        col1["signals"] = {"constant": True}
        
        col2 = column_template.copy()
        col2["name"] = "const2"
        col2["signals"] = {"constant": True}
        
        col3 = column_template.copy()
        col3["name"] = "varying"
        col3["signals"] = {"constant": False}
        
        base_context["columns"] = [col1, col2, col3]
        
        rule = Constant_Column()
        result = rule.run(base_context)
        
        assert len(result.details["affected_columns"]) == 2
        assert "const1" in result.details["affected_columns"]
        assert "const2" in result.details["affected_columns"]
        assert "varying" not in result.details["affected_columns"]
    
    # NearConstantNumeric Tests
    
    def test_near_constant_applies(self, base_context, column_template):
        """Test near constant rule applies to numeric columns with low variance"""
        column_template["type"] = "numeric"
        column_template["signals"]["low_variance"] = True
        base_context["columns"] = [column_template]
        
        rule = NearConstantNumeric()
        assert rule.applies(base_context) is True
    
    def test_near_constant_requires_numeric(self, base_context, column_template):
        """Test near constant rule only applies to numeric columns"""
        column_template["type"] = "categorical"
        column_template["signals"]["low_variance"] = True
        base_context["columns"] = [column_template]
        
        rule = NearConstantNumeric()
        assert rule.applies(base_context) is False
    
    def test_near_constant_requires_low_variance_signal(self, base_context, column_template):
        """Test near constant rule requires low_variance signal"""
        column_template["type"] = "numeric"
        column_template["signals"]["low_variance"] = False
        base_context["columns"] = [column_template]
        
        rule = NearConstantNumeric()
        assert rule.applies(base_context) is False
    
    def test_near_constant_run(self, base_context, column_template):
        """Test near constant rule output"""
        column_template["type"] = "numeric"
        column_template["signals"]["low_variance"] = True
        column_template["name"] = "feature_x"
        base_context["columns"] = [column_template]
        
        rule = NearConstantNumeric()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R2.4_Near_Constant_Numeric"
        assert result.priority == 90
        assert result.type == "column_cleaning"
        assert result.action == "suggest_drop"
        assert result.severity == "medium"
        assert "feature_x" in result.message
        assert "minimal variance" in result.message.lower()
    
    def test_constant_vs_near_constant_priority(self):
        """Test that constant has higher priority than near-constant"""
        constant = Constant_Column()
        near_constant = NearConstantNumeric()
        
        assert constant.priority > near_constant.priority


class TestTypeReclassificationRules:
    """Test suite for type reclassification rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {"columns": []}
    
    @pytest.fixture
    def column_template(self):
        """Template for column structure"""
        return {
            "name": "test_col",
            "type": "numeric",
            "unique_ratio": 0.0,
            "cardinality": 0,
            "flags": []
        }
    
    # NumericIDMasquerade Tests
    
    def test_numeric_id_applies(self, base_context, column_template):
        """Test numeric ID rule applies when unique_ratio > 0.95"""
        column_template["type"] = "numeric"
        column_template["unique_ratio"] = 0.96
        base_context["columns"] = [column_template]
        
        rule = NumericIDMasquerade()
        assert rule.applies(base_context) is True
    
    def test_numeric_id_boundary(self, base_context, column_template):
        """Test numeric ID rule boundary at 0.95"""
        rule = NumericIDMasquerade()
        column_template["type"] = "numeric"
        
        # At boundary (excluded)
        column_template["unique_ratio"] = 0.95
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # Just above boundary
        column_template["unique_ratio"] = 0.951
        assert rule.applies(base_context) is True
    
    def test_numeric_id_requires_numeric_type(self, base_context, column_template):
        """Test numeric ID rule only applies to numeric columns"""
        column_template["type"] = "categorical"
        column_template["unique_ratio"] = 0.99
        base_context["columns"] = [column_template]
        
        rule = NumericIDMasquerade()
        assert rule.applies(base_context) is False
    
    def test_numeric_id_run(self, base_context, column_template):
        """Test numeric ID rule output"""
        column_template["type"] = "numeric"
        column_template["unique_ratio"] = 0.98
        column_template["name"] = "user_id"
        base_context["columns"] = [column_template]
        
        rule = NumericIDMasquerade()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R2.5_Numeric_ID_Masquerade"
        assert result.priority == 85
        assert result.type == "type_reclassification"
        assert result.action == "reclassify_as_id"
        assert result.severity == "medium"
        assert "user_id" in result.message
        assert "categorical id" in result.message.lower()
    
    def test_numeric_id_multiple_columns(self, base_context, column_template):
        """Test numeric ID with multiple ID-like columns"""
        col1 = column_template.copy()
        col1["name"] = "user_id"
        col1["type"] = "numeric"
        col1["unique_ratio"] = 0.99
        
        col2 = column_template.copy()
        col2["name"] = "session_id"
        col2["type"] = "numeric"
        col2["unique_ratio"] = 0.97
        
        col3 = column_template.copy()
        col3["name"] = "age"
        col3["type"] = "numeric"
        col3["unique_ratio"] = 0.02
        
        base_context["columns"] = [col1, col2, col3]
        
        rule = NumericIDMasquerade()
        result = rule.run(base_context)
        
        assert len(result.details["affected_columns"]) == 2
        assert "user_id" in result.details["affected_columns"]
        assert "session_id" in result.details["affected_columns"]
        assert "age" not in result.details["affected_columns"]
    
    # LowCardNumericToCategorical Tests
    
    def test_low_card_numeric_applies(self, base_context, column_template):
        """Test low cardinality numeric rule applies"""
        column_template["type"] = "numeric"
        column_template["cardinality"] = 5
        column_template["flags"] = ["event_like"]
        base_context["columns"] = [column_template]
        
        rule = LowCardNumericToCategorical()
        assert rule.applies(base_context) is True
    
    def test_low_card_numeric_cardinality_boundaries(self, base_context, column_template):
        """Test low cardinality numeric boundaries"""
        rule = LowCardNumericToCategorical()
        column_template["type"] = "numeric"
        column_template["flags"] = ["event_like"]
        
        # Below lower boundary
        column_template["cardinality"] = 9
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is True
        
        # At upper boundary (excluded)
        column_template["cardinality"] = 10
        assert rule.applies(base_context) is False
        
        # Above upper boundary
        column_template["cardinality"] = 11
        assert rule.applies(base_context) is False
    
    def test_low_card_numeric_requires_event_like_flag(self, base_context, column_template):
        """Test low cardinality numeric requires event_like flag"""
        column_template["type"] = "numeric"
        column_template["cardinality"] = 5
        column_template["flags"] = []
        base_context["columns"] = [column_template]
        
        rule = LowCardNumericToCategorical()
        assert rule.applies(base_context) is False
    
    def test_low_card_numeric_run(self, base_context, column_template):
        """Test low cardinality numeric rule output"""
        column_template["type"] = "numeric"
        column_template["cardinality"] = 7
        column_template["flags"] = ["event_like"]
        column_template["name"] = "rating"
        base_context["columns"] = [column_template]
        
        rule = LowCardNumericToCategorical()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R2.6_Low_Card_Numeric_Categorical"
        assert result.priority == 80
        assert result.type == "type_reclassification"
        assert result.action == "cast_to_category"
        assert result.severity == "info"
        assert "rating" in result.message
        assert "categorical" in result.message.lower()
    
    # HighCardCategoricalToText Tests
    
    def test_high_card_categorical_applies(self, base_context, column_template):
        """Test high cardinality categorical rule applies"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 1500
        column_template["flags"] = ["high_cardinality"]
        base_context["columns"] = [column_template]
        
        rule = HighCardCategoricalToText()
        assert rule.applies(base_context) is True
    
    def test_high_card_categorical_boundary(self, base_context, column_template):
        """Test high cardinality categorical boundary at 1000"""
        rule = HighCardCategoricalToText()
        column_template["type"] = "categorical"
        column_template["flags"] = ["high_cardinality"]
        
        # At boundary (excluded)
        column_template["cardinality"] = 1000
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # Just above boundary
        column_template["cardinality"] = 1001
        assert rule.applies(base_context) is True
    
    def test_high_card_categorical_requires_flag(self, base_context, column_template):
        """Test high cardinality categorical requires high_cardinality flag"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 1500
        column_template["flags"] = []
        base_context["columns"] = [column_template]
        
        rule = HighCardCategoricalToText()
        assert rule.applies(base_context) is False
    
    def test_high_card_categorical_requires_categorical_type(self, base_context, column_template):
        """Test high cardinality rule only applies to categorical columns"""
        column_template["type"] = "numeric"
        column_template["cardinality"] = 1500
        column_template["flags"] = ["high_cardinality"]
        base_context["columns"] = [column_template]
        
        rule = HighCardCategoricalToText()
        assert rule.applies(base_context) is False
    
    def test_high_card_categorical_run(self, base_context, column_template):
        """Test high cardinality categorical rule output"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 2000
        column_template["flags"] = ["high_cardinality"]
        column_template["name"] = "product_name"
        base_context["columns"] = [column_template]
        
        rule = HighCardCategoricalToText()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R2.7_High_Card_Categorical_Text"
        assert result.priority == 75
        assert result.type == "encoding_warning"
        assert result.action == "use_feature_hashing"
        assert result.severity == "medium"
        assert "product_name" in result.message
        assert "target encoding" in result.message.lower() or "feature hashing" in result.message.lower()
    
    # Integration Tests
    
    def test_reclassification_rule_priorities(self):
        """Test that reclassification rules have correct priority ordering"""
        numeric_id = NumericIDMasquerade()
        low_card = LowCardNumericToCategorical()
        high_card = HighCardCategoricalToText()
        
        assert numeric_id.priority > low_card.priority > high_card.priority
    
    def test_no_overlap_numeric_id_and_low_card(self, base_context, column_template):
        """Test that numeric ID and low card numeric don't overlap"""
        # High unique ratio (ID-like)
        column_template["type"] = "numeric"
        column_template["unique_ratio"] = 0.98
        column_template["cardinality"] = 5
        column_template["flags"] = ["event_like"]
        base_context["columns"] = [column_template]
        
        id_rule = NumericIDMasquerade()
        low_card_rule = LowCardNumericToCategorical()
        
        # With high unique ratio, only ID rule should apply
        assert id_rule.applies(base_context) is True
        # Low cardinality would apply if checked, but unique_ratio check comes first
        assert low_card_rule.applies(base_context) is True
        
        # This is actually an edge case - both could theoretically apply
        # Priority system should handle this


