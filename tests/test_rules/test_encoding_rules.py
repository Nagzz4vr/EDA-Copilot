import pytest
from typing import Dict, Any
from signals.rules.base_rule import RuleOutput
from signals.rules.feature_engineering.encoding import (
    BinaryCategorical,
    LowCardOneHot,
    ImbalancedCategoricalGrouping,

)
from signals.rules.feature_engineering.normalization import(
    HighlySkewedNumeric)
from signals.rules.feature_engineering.text_processing import(
    LongTextNLP,
    ShortTextAsCategorical)


class TestCategoricalEncodingRules:
    """Test suite for categorical encoding rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {"columns": []}
    
    @pytest.fixture
    def column_template(self):
        """Template for column structure"""
        return {
            "name": "test_col",
            "type": "categorical",
            "cardinality": 0,
            "flags": [],
            "stats": {}
        }
    
    # BinaryCategorical Tests
    
    def test_binary_categorical_applies(self, base_context, column_template):
        """Test binary categorical rule applies with cardinality=2"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 2
        base_context["columns"] = [column_template]
        
        rule = BinaryCategorical()
        assert rule.applies(base_context) is True
    
    def test_binary_categorical_does_not_apply_cardinality_1(self, base_context, column_template):
        """Test binary categorical doesn't apply with cardinality=1"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 1
        base_context["columns"] = [column_template]
        
        rule = BinaryCategorical()
        assert rule.applies(base_context) is False
    
    def test_binary_categorical_does_not_apply_cardinality_3(self, base_context, column_template):
        """Test binary categorical doesn't apply with cardinality=3"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 3
        base_context["columns"] = [column_template]
        
        rule = BinaryCategorical()
        assert rule.applies(base_context) is False
    
    def test_binary_categorical_requires_categorical_type(self, base_context, column_template):
        """Test binary categorical only applies to categorical type"""
        column_template["type"] = "numeric"
        column_template["cardinality"] = 2
        base_context["columns"] = [column_template]
        
        rule = BinaryCategorical()
        assert rule.applies(base_context) is False
    
    def test_binary_categorical_run(self, base_context, column_template):
        """Test binary categorical rule output"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 2
        column_template["name"] = "gender"
        base_context["columns"] = [column_template]
        
        rule = BinaryCategorical()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R3.3_Binary_Categorical"
        assert result.priority == 60
        assert result.type == "encoding_strategy"
        assert result.action == "label_encode"
        assert result.severity == "info"
        assert "gender" in result.message
        assert "label encoding" in result.message.lower()
        assert result.details["affected_columns"] == ["gender"]
    
    def test_binary_categorical_multiple_columns(self, base_context, column_template):
        """Test binary categorical with multiple binary columns"""
        col1 = column_template.copy()
        col1["name"] = "is_active"
        col1["cardinality"] = 2
        
        col2 = column_template.copy()
        col2["name"] = "has_subscription"
        col2["cardinality"] = 2
        
        col3 = column_template.copy()
        col3["name"] = "category"
        col3["cardinality"] = 5
        
        base_context["columns"] = [col1, col2, col3]
        
        rule = BinaryCategorical()
        result = rule.run(base_context)
        
        assert len(result.details["affected_columns"]) == 2
        assert "is_active" in result.details["affected_columns"]
        assert "has_subscription" in result.details["affected_columns"]
    
    # LowCardOneHot Tests
    
    def test_low_card_one_hot_applies(self, base_context, column_template):
        """Test low card one-hot rule applies with cardinality 3-10"""
        rule = LowCardOneHot()
        column_template["type"] = "categorical"
        
        for cardinality in [3, 5, 7, 10]:
            column_template["cardinality"] = cardinality
            base_context["columns"] = [column_template]
            assert rule.applies(base_context) is True, f"Failed for cardinality {cardinality}"
    
    def test_low_card_one_hot_boundaries(self, base_context, column_template):
        """Test low card one-hot rule boundaries"""
        rule = LowCardOneHot()
        column_template["type"] = "categorical"
        
        # Below lower boundary
        column_template["cardinality"] = 2
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # At lower boundary (included)
        column_template["cardinality"] = 3
        assert rule.applies(base_context) is True
        
        # At upper boundary (included)
        column_template["cardinality"] = 10
        assert rule.applies(base_context) is True
        
        # Above upper boundary
        column_template["cardinality"] = 11
        assert rule.applies(base_context) is False
    
    def test_low_card_one_hot_run(self, base_context, column_template):
        """Test low card one-hot rule output"""
        column_template["type"] = "categorical"
        column_template["cardinality"] = 5
        column_template["name"] = "color"
        base_context["columns"] = [column_template]
        
        rule = LowCardOneHot()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R3.4_Low_Card_One_Hot"
        assert result.priority == 55
        assert result.type == "encoding_strategy"
        assert result.action == "one_hot_encode"
        assert result.severity == "info"
        assert "color" in result.message
        assert "one-hot" in result.message.lower()
    
    # ImbalancedCategoricalGrouping Tests
    
    def test_imbalanced_categorical_applies(self, base_context, column_template):
        """Test imbalanced categorical rule applies with flag and stats"""
        column_template["type"] = "categorical"
        column_template["flags"] = ["imbalanced"]
        column_template["stats"]["imbalanced"] = True
        base_context["columns"] = [column_template]
        
        rule = ImbalancedCategoricalGrouping()
        assert rule.applies(base_context) is True
    
    def test_imbalanced_categorical_requires_flag(self, base_context, column_template):
        """Test imbalanced categorical requires imbalanced flag"""
        column_template["type"] = "categorical"
        column_template["flags"] = []
        column_template["stats"]["imbalanced"] = True
        base_context["columns"] = [column_template]
        
        rule = ImbalancedCategoricalGrouping()
        assert rule.applies(base_context) is False
    
    def test_imbalanced_categorical_requires_stats(self, base_context, column_template):
        """Test imbalanced categorical requires stats to be True"""
        column_template["type"] = "categorical"
        column_template["flags"] = ["imbalanced"]
        column_template["stats"]["imbalanced"] = False
        base_context["columns"] = [column_template]
        
        rule = ImbalancedCategoricalGrouping()
        assert rule.applies(base_context) is False
    
    def test_imbalanced_categorical_run(self, base_context, column_template):
        """Test imbalanced categorical rule output"""
        column_template["type"] = "categorical"
        column_template["flags"] = ["imbalanced"]
        column_template["stats"]["imbalanced"] = True
        column_template["name"] = "country"
        base_context["columns"] = [column_template]
        
        rule = ImbalancedCategoricalGrouping()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R3.5_Imbalanced_Categorical"
        assert result.priority == 50
        assert result.type == "preprocessing_suggestion"
        assert result.action == "group_rare_categories"
        assert result.severity == "medium"
        assert "country" in result.message
        assert "other" in result.message.lower() or "group" in result.message.lower()
        assert result.details["threshold"] == "90%"
    
    # Integration Tests
    
    def test_encoding_rule_priorities(self):
        """Test that encoding rules have correct priority ordering"""
        binary = BinaryCategorical()
        low_card = LowCardOneHot()
        imbalanced = ImbalancedCategoricalGrouping()
        
        assert binary.priority > low_card.priority > imbalanced.priority
    
    def test_no_overlap_binary_and_low_card(self, base_context, column_template):
        """Test that binary and low card rules don't overlap"""
        column_template["type"] = "categorical"
        
        # Binary (cardinality=2)
        column_template["cardinality"] = 2
        base_context["columns"] = [column_template]
        
        binary_rule = BinaryCategorical()
        low_card_rule = LowCardOneHot()
        
        assert binary_rule.applies(base_context) is True
        assert low_card_rule.applies(base_context) is False


class TestNumericPreprocessingRules:
    """Test suite for numeric preprocessing rules"""
    
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
            "stats": {
                "skew": 0.0,
                "min": 0.0
            }
        }
    
    # HighlySkewedNumeric Tests
    
    def test_highly_skewed_applies(self, base_context, column_template):
        """Test highly skewed rule applies with all conditions met"""
        column_template["type"] = "numeric"
        column_template["signals"]["skewed"] = True
        column_template["stats"]["skew"] = 2.0
        column_template["stats"]["min"] = 1.0
        base_context["columns"] = [column_template]
        
        rule = HighlySkewedNumeric()
        assert rule.applies(base_context) is True
    
    def test_highly_skewed_boundary_skew(self, base_context, column_template):
        """Test highly skewed rule boundary for skew value"""
        rule = HighlySkewedNumeric()
        column_template["type"] = "numeric"
        column_template["signals"]["skewed"] = True
        column_template["stats"]["min"] = 1.0
        
        # At boundary (excluded)
        column_template["stats"]["skew"] = 1.5
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # Just above boundary
        column_template["stats"]["skew"] = 1.51
        assert rule.applies(base_context) is True
    
    def test_highly_skewed_requires_positive_min(self, base_context, column_template):
        """Test highly skewed rule requires min > 0"""
        rule = HighlySkewedNumeric()
        column_template["type"] = "numeric"
        column_template["signals"]["skewed"] = True
        column_template["stats"]["skew"] = 2.0
        
        # min = 0 (excluded)
        column_template["stats"]["min"] = 0
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # min > 0 (included)
        column_template["stats"]["min"] = 0.1
        assert rule.applies(base_context) is True
    
    def test_highly_skewed_requires_signal(self, base_context, column_template):
        """Test highly skewed rule requires skewed signal"""
        column_template["type"] = "numeric"
        column_template["signals"]["skewed"] = False
        column_template["stats"]["skew"] = 2.0
        column_template["stats"]["min"] = 1.0
        base_context["columns"] = [column_template]
        
        rule = HighlySkewedNumeric()
        assert rule.applies(base_context) is False
    
    def test_highly_skewed_run(self, base_context, column_template):
        """Test highly skewed rule output"""
        column_template["type"] = "numeric"
        column_template["signals"]["skewed"] = True
        column_template["stats"]["skew"] = 3.5
        column_template["stats"]["min"] = 5.0
        column_template["name"] = "income"
        base_context["columns"] = [column_template]
        
        rule = HighlySkewedNumeric()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R3.1_Highly_Skewed_Numeric"
        assert result.priority == 85
        assert result.type == "normalization_recommendation"
        assert result.action == "log_transform"
        assert result.severity == "medium"
        assert "income" in result.message
        assert "log" in result.message.lower()
        assert result.details["transformation"] == "np.log1p"
    
    def test_highly_skewed_multiple_columns(self, base_context, column_template):
        """Test highly skewed with multiple skewed columns"""
        col1 = column_template.copy()
        col1["name"] = "price"
        col1["signals"] = {"skewed": True}
        col1["stats"] = {"skew": 2.5, "min": 10.0}
        
        col2 = column_template.copy()
        col2["name"] = "revenue"
        col2["signals"] = {"skewed": True}
        col2["stats"] = {"skew": 4.0, "min": 100.0}
        
        col3 = column_template.copy()
        col3["name"] = "age"
        col3["signals"] = {"skewed": False}
        col3["stats"] = {"skew": 0.5, "min": 18.0}
        
        base_context["columns"] = [col1, col2, col3]
        
        rule = HighlySkewedNumeric()
        result = rule.run(base_context)
        
        assert len(result.details["affected_columns"]) == 2
        assert "price" in result.details["affected_columns"]
        assert "revenue" in result.details["affected_columns"]


class TestTextPreprocessingRules:
    """Test suite for text preprocessing rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {"columns": []}
    
    @pytest.fixture
    def column_template(self):
        """Template for column structure"""
        return {
            "name": "test_col",
            "type": "text",
            "flags": [],
            "stats": {
                "avg_length": 0
            },
            "cardinality": 0
        }
    
    # LongTextNLP Tests
    
    def test_long_text_applies(self, base_context, column_template):
        """Test long text NLP rule applies with all conditions met"""
        column_template["type"] = "text"
        column_template["flags"] = ["long_text"]
        column_template["stats"]["avg_length"] = 150
        base_context["columns"] = [column_template]
        
        rule = LongTextNLP()
        assert rule.applies(base_context) is True
    
    def test_long_text_boundary_avg_length(self, base_context, column_template):
        """Test long text rule boundary for avg_length"""
        rule = LongTextNLP()
        column_template["type"] = "text"
        column_template["flags"] = ["long_text"]
        
        # At boundary (excluded)
        column_template["stats"]["avg_length"] = 100
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # Just above boundary
        column_template["stats"]["avg_length"] = 101
        assert rule.applies(base_context) is True
    
    def test_long_text_requires_flag(self, base_context, column_template):
        """Test long text rule requires long_text flag"""
        column_template["type"] = "text"
        column_template["flags"] = []
        column_template["stats"]["avg_length"] = 150
        base_context["columns"] = [column_template]
        
        rule = LongTextNLP()
        assert rule.applies(base_context) is False
    
    def test_long_text_run(self, base_context, column_template):
        """Test long text rule output"""
        column_template["type"] = "text"
        column_template["flags"] = ["long_text"]
        column_template["stats"]["avg_length"] = 250
        column_template["name"] = "review_text"
        base_context["columns"] = [column_template]
        
        rule = LongTextNLP()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R3.6_Long_Text_NLP"
        assert result.priority == 70
        assert result.type == "preprocessing_pipeline"
        assert result.action == "route_to_nlp"
        assert result.severity == "medium"
        assert "review_text" in result.message
        assert "nlp" in result.message.lower()
    
    # ShortTextAsCategorical Tests
    
    def test_short_text_applies(self, base_context, column_template):
        """Test short text as categorical rule applies"""
        column_template["type"] = "text"
        column_template["stats"]["avg_length"] = 15
        column_template["cardinality"] = 50
        base_context["columns"] = [column_template]
        
        rule = ShortTextAsCategorical()
        assert rule.applies(base_context) is True
    
    def test_short_text_boundary_avg_length(self, base_context, column_template):
        """Test short text rule boundary for avg_length"""
        rule = ShortTextAsCategorical()
        column_template["type"] = "text"
        column_template["cardinality"] = 50
        
        # Below boundary (included)
        column_template["stats"]["avg_length"] = 19
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is True
        
        # At boundary (excluded)
        column_template["stats"]["avg_length"] = 20
        assert rule.applies(base_context) is False
    
    def test_short_text_boundary_cardinality(self, base_context, column_template):
        """Test short text rule boundary for cardinality"""
        rule = ShortTextAsCategorical()
        column_template["type"] = "text"
        column_template["stats"]["avg_length"] = 15
        
        # Below boundary (included)
        column_template["cardinality"] = 99
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is True
        
        # At boundary (excluded)
        column_template["cardinality"] = 100
        assert rule.applies(base_context) is False
    
    def test_short_text_run(self, base_context, column_template):
        """Test short text rule output"""
        column_template["type"] = "text"
        column_template["stats"]["avg_length"] = 10
        column_template["cardinality"] = 30
        column_template["name"] = "status_code"
        base_context["columns"] = [column_template]
        
        rule = ShortTextAsCategorical()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R3.7_Short_Text_Categorical"
        assert result.priority == 75
        assert result.type == "type_reclassification"
        assert result.action == "cast_to_categorical"
        assert result.severity == "info"
        assert "status_code" in result.message
        assert "categorical" in result.message.lower()
    
    # Integration Tests
    
    def test_text_rule_priorities(self):
        """Test that text rules have correct priority ordering"""
        short_text = ShortTextAsCategorical()
        long_text = LongTextNLP()
        
        assert short_text.priority > long_text.priority
    
    def test_no_overlap_long_and_short_text(self, base_context, column_template):
        """Test that long and short text rules don't overlap"""
        column_template["type"] = "text"
        column_template["cardinality"] = 50
        
        # Short text scenario
        column_template["stats"]["avg_length"] = 15
        column_template["flags"] = []
        base_context["columns"] = [column_template]
        
        short_rule = ShortTextAsCategorical()
        long_rule = LongTextNLP()
        
        assert short_rule.applies(base_context) is True
        assert long_rule.applies(base_context) is False
        
        # Long text scenario
        column_template["stats"]["avg_length"] = 150
        column_template["flags"] = ["long_text"]
        
        assert short_rule.applies(base_context) is False
        assert long_rule.applies(base_context) is True


