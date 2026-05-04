import pytest
from typing import Dict, Any
from signals.rules.base_rule import RuleOutput
from signals.rules.data_quality.deduplication import (
    CriticalDuplicateLevel,
    HighDuplicateThreshold,
    NoDuplicates,
    LowDuplicateNoise,

)
from signals.rules.data_quality.missing_data import(
    HighMissingRule,
    ModerateMissingRule,
    PatternMissingRule,
    RandomMissingRule)


class TestDuplicateRules:
    """Test suite for duplicate detection rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {
            "dataset_health": {},
            "columns": []
        }
    
    # CriticalDuplicateLevel Tests
    
    def test_critical_duplicate_applies(self, base_context):
        """Test critical duplicate rule applies when duplicates > 20%"""
        base_context["dataset_health"]["duplicate_percent"] = 25
        rule = CriticalDuplicateLevel()
        
        assert rule.applies(base_context) is True
    
    def test_critical_duplicate_does_not_apply(self, base_context):
        """Test critical duplicate rule doesn't apply when duplicates <= 20%"""
        base_context["dataset_health"]["duplicate_percent"] = 20
        rule = CriticalDuplicateLevel()
        
        assert rule.applies(base_context) is False
    
    def test_critical_duplicate_run(self, base_context):
        """Test critical duplicate rule output"""
        base_context["dataset_health"]["duplicate_percent"] = 30
        rule = CriticalDuplicateLevel()
        
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.2_critical_duplicates"
        assert result.priority == 100
        assert result.type == "dataset"
        assert result.action == "block_analysis"
        assert result.severity == "critical"
        assert "30%" in result.message
        assert result.details["duplicate_percent"] == 30
    
    def test_critical_duplicate_edge_case_21_percent(self, base_context):
        """Test critical duplicate at edge case 21%"""
        base_context["dataset_health"]["duplicate_percent"] = 21
        rule = CriticalDuplicateLevel()
        
        assert rule.applies(base_context) is True
    
    # HighDuplicateThreshold Tests
    
    def test_high_duplicate_applies(self, base_context):
        """Test high duplicate rule applies when 5% < duplicates <= 20%"""
        base_context["dataset_health"]["duplicate_percent"] = 15
        rule = HighDuplicateThreshold()
        
        assert rule.applies(base_context) is True
    
    def test_high_duplicate_boundaries(self, base_context):
        """Test high duplicate rule boundaries"""
        rule = HighDuplicateThreshold()
        
        # Lower boundary (excluded)
        base_context["dataset_health"]["duplicate_percent"] = 5
        assert rule.applies(base_context) is False
        
        # Just above lower boundary
        base_context["dataset_health"]["duplicate_percent"] = 5.1
        assert rule.applies(base_context) is True
        
        # Upper boundary (included)
        base_context["dataset_health"]["duplicate_percent"] = 20
        assert rule.applies(base_context) is True
        
        # Above upper boundary
        base_context["dataset_health"]["duplicate_percent"] = 20.1
        assert rule.applies(base_context) is False
    
    def test_high_duplicate_run(self, base_context):
        """Test high duplicate rule output"""
        base_context["dataset_health"]["duplicate_percent"] = 10
        rule = HighDuplicateThreshold()
        
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.1_high_duplicates"
        assert result.priority == 50
        assert result.action == "suggest_deduplication"
        assert result.severity == "medium"
        assert "10%" in result.message
    
    # NoDuplicates Tests
    
    def test_no_duplicates_applies(self, base_context):
        """Test no duplicates rule applies when duplicates = 0"""
        base_context["dataset_health"]["duplicate_percent"] = 0
        rule = NoDuplicates()
        
        assert rule.applies(base_context) is True
    
    def test_no_duplicates_does_not_apply(self, base_context):
        """Test no duplicates rule doesn't apply when duplicates > 0"""
        base_context["dataset_health"]["duplicate_percent"] = 0.1
        rule = NoDuplicates()
        
        assert rule.applies(base_context) is False
    
    def test_no_duplicates_run(self, base_context):
        """Test no duplicates rule output"""
        base_context["dataset_health"]["duplicate_percent"] = 0
        rule = NoDuplicates()
        
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.3_no_duplicates"
        assert result.priority == 10
        assert result.action == "skip_deduplication"
        assert result.severity == "info"
        assert result.details["duplicate_percent"] == 0
    
    # LowDuplicateNoise Tests
    
    def test_low_duplicate_applies(self, base_context):
        """Test low duplicate rule applies when 0% < duplicates <= 5%"""
        base_context["dataset_health"]["duplicate_percent"] = 3
        rule = LowDuplicateNoise()
        
        assert rule.applies(base_context) is True
    
    def test_low_duplicate_boundaries(self, base_context):
        """Test low duplicate rule boundaries"""
        rule = LowDuplicateNoise()
        
        # Zero (excluded)
        base_context["dataset_health"]["duplicate_percent"] = 0
        assert rule.applies(base_context) is False
        
        # Just above zero
        base_context["dataset_health"]["duplicate_percent"] = 0.1
        assert rule.applies(base_context) is True
        
        # Upper boundary (included)
        base_context["dataset_health"]["duplicate_percent"] = 5
        assert rule.applies(base_context) is True
        
        # Above upper boundary
        base_context["dataset_health"]["duplicate_percent"] = 5.1
        assert rule.applies(base_context) is False
    
    def test_low_duplicate_run(self, base_context):
        """Test low duplicate rule output"""
        base_context["dataset_health"]["duplicate_percent"] = 2.5
        rule = LowDuplicateNoise()
        
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.4_low_duplicate_noise"
        assert result.priority == 5
        assert result.action == "ignore_duplicates"
        assert result.severity == "low"
        assert "2.5%" in result.message
    
    # Integration Test - Duplicate Rule Priorities
    
    def test_duplicate_rule_priorities(self):
        """Test that duplicate rules have correct priority ordering"""
        critical = CriticalDuplicateLevel()
        high = HighDuplicateThreshold()
        no_dup = NoDuplicates()
        low = LowDuplicateNoise()
        
        assert critical.priority > high.priority > no_dup.priority > low.priority


class TestMissingValueRules:
    """Test suite for missing value detection rules"""
    
    @pytest.fixture
    def base_context(self):
        """Base context structure"""
        return {
            "columns": []
        }
    
    @pytest.fixture
    def column_template(self):
        """Template for column structure"""
        return {
            "name": "test_col",
            "type": "numeric",
            "signals": {},
            "missing": {"percent": 0},
            "missing_pattern": {
                "transitions": 0,
                "max_consecutive_missing": 0
            }
        }
    
    # HighMissingRule Tests
    
    def test_high_missing_applies(self, base_context, column_template):
        """Test high missing rule applies when column has high_missing signal"""
        column_template["signals"]["high_missing"] = True
        base_context["columns"] = [column_template]
        
        rule = HighMissingRule()
        assert rule.applies(base_context) is True
    
    def test_high_missing_does_not_apply(self, base_context, column_template):
        """Test high missing rule doesn't apply without signal"""
        column_template["signals"]["high_missing"] = False
        base_context["columns"] = [column_template]
        
        rule = HighMissingRule()
        assert rule.applies(base_context) is False
    
    def test_high_missing_run(self, base_context, column_template):
        """Test high missing rule output"""
        column_template["signals"]["high_missing"] = True
        column_template["name"] = "salary"
        base_context["columns"] = [column_template]
        
        rule = HighMissingRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.5_high_missing"
        assert result.priority == 100
        assert result.type == "column_health"
        assert result.action == "warning"
        assert result.severity == "high"
        assert "salary" in result.message
        assert result.details["affected_columns"] == ["salary"]
    
    def test_high_missing_multiple_columns(self, base_context, column_template):
        """Test high missing with multiple affected columns"""
        col1 = column_template.copy()
        col1["name"] = "col1"
        col1["signals"] = {"high_missing": True}
        
        col2 = column_template.copy()
        col2["name"] = "col2"
        col2["signals"] = {"high_missing": True}
        
        col3 = column_template.copy()
        col3["name"] = "col3"
        col3["signals"] = {"high_missing": False}
        
        base_context["columns"] = [col1, col2, col3]
        
        rule = HighMissingRule()
        result = rule.run(base_context)
        
        assert len(result.details["affected_columns"]) == 2
        assert "col1" in result.details["affected_columns"]
        assert "col2" in result.details["affected_columns"]
        assert "col3" not in result.details["affected_columns"]
    
    # ModerateMissingRule Tests
    
    def test_moderate_missing_applies_numeric(self, base_context, column_template):
        """Test moderate missing rule applies to numeric columns"""
        column_template["type"] = "numeric"
        column_template["signals"]["moderate_missing"] = True
        base_context["columns"] = [column_template]
        
        rule = ModerateMissingRule()
        assert rule.applies(base_context) is True
    
    def test_moderate_missing_applies_categorical(self, base_context, column_template):
        """Test moderate missing rule applies to categorical columns"""
        column_template["type"] = "categorical"
        column_template["signals"]["moderate_missing"] = True
        base_context["columns"] = [column_template]
        
        rule = ModerateMissingRule()
        assert rule.applies(base_context) is True
    
    def test_moderate_missing_does_not_apply_text(self, base_context, column_template):
        """Test moderate missing rule doesn't apply to text columns"""
        column_template["type"] = "text"
        column_template["signals"]["moderate_missing"] = True
        base_context["columns"] = [column_template]
        
        rule = ModerateMissingRule()
        assert rule.applies(base_context) is False
    
    def test_moderate_missing_run(self, base_context, column_template):
        """Test moderate missing rule output"""
        column_template["type"] = "numeric"
        column_template["signals"]["moderate_missing"] = True
        column_template["name"] = "age"
        base_context["columns"] = [column_template]
        
        rule = ModerateMissingRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.6_moderate_missing"
        assert result.priority == 50
        assert result.type == "column_health"
        assert result.severity == "medium"
        assert "age" in result.message
    
    # PatternMissingRule Tests
    
    def test_pattern_missing_applies(self, base_context, column_template):
        """Test pattern missing rule applies with low transitions and high consecutive missing"""
        column_template["missing_pattern"]["transitions"] = 3
        column_template["missing_pattern"]["max_consecutive_missing"] = 150
        base_context["columns"] = [column_template]
        
        rule = PatternMissingRule()
        assert rule.applies(base_context) is True
    
    def test_pattern_missing_boundary_transitions(self, base_context, column_template):
        """Test pattern missing rule boundary for transitions"""
        rule = PatternMissingRule()
        
        # transitions = 5 (boundary, excluded)
        column_template["missing_pattern"]["transitions"] = 5
        column_template["missing_pattern"]["max_consecutive_missing"] = 150
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # transitions = 4 (included)
        column_template["missing_pattern"]["transitions"] = 4
        assert rule.applies(base_context) is True
    
    def test_pattern_missing_boundary_consecutive(self, base_context, column_template):
        """Test pattern missing rule boundary for consecutive missing"""
        rule = PatternMissingRule()
        
        # max_consecutive_missing = 100 (boundary, excluded)
        column_template["missing_pattern"]["transitions"] = 3
        column_template["missing_pattern"]["max_consecutive_missing"] = 100
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # max_consecutive_missing = 101 (included)
        column_template["missing_pattern"]["max_consecutive_missing"] = 101
        assert rule.applies(base_context) is True
    
    def test_pattern_missing_run(self, base_context, column_template):
        """Test pattern missing rule output"""
        column_template["missing_pattern"]["transitions"] = 2
        column_template["missing_pattern"]["max_consecutive_missing"] = 200
        column_template["name"] = "sensor_reading"
        base_context["columns"] = [column_template]
        
        rule = PatternMissingRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.7_pattern_missing"
        assert result.priority == 10
        assert result.action == "investigate"
        assert result.severity == "medium"
        assert "sensor_reading" in result.message
        assert "blocks of NAs" in result.message.lower() or "non-random" in result.message.lower()
    
    # RandomMissingRule Tests
    
    def test_random_missing_applies(self, base_context, column_template):
        """Test random missing rule applies with low percent and high transitions"""
        column_template["missing"]["percent"] = 3
        column_template["missing_pattern"]["transitions"] = 60
        base_context["columns"] = [column_template]
        
        rule = RandomMissingRule()
        assert rule.applies(base_context) is True
    
    def test_random_missing_boundary_percent(self, base_context, column_template):
        """Test random missing rule boundary for percent"""
        rule = RandomMissingRule()
        column_template["missing_pattern"]["transitions"] = 60
        
        # percent = 5 (boundary, excluded)
        column_template["missing"]["percent"] = 5
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # percent = 4.9 (included)
        column_template["missing"]["percent"] = 4.9
        assert rule.applies(base_context) is True
    
    def test_random_missing_boundary_transitions(self, base_context, column_template):
        """Test random missing rule boundary for transitions"""
        rule = RandomMissingRule()
        column_template["missing"]["percent"] = 3
        
        # transitions = 50 (boundary, excluded)
        column_template["missing_pattern"]["transitions"] = 50
        base_context["columns"] = [column_template]
        assert rule.applies(base_context) is False
        
        # transitions = 51 (included)
        column_template["missing_pattern"]["transitions"] = 51
        assert rule.applies(base_context) is True
    
    def test_random_missing_run(self, base_context, column_template):
        """Test random missing rule output"""
        column_template["missing"]["percent"] = 2
        column_template["missing_pattern"]["transitions"] = 80
        column_template["name"] = "optional_field"
        base_context["columns"] = [column_template]
        
        rule = RandomMissingRule()
        result = rule.run(base_context)
        
        assert isinstance(result, RuleOutput)
        assert result.rule_name == "R1.8_random_missing"
        assert result.priority == 5
        assert result.action == "safe_to_drop_rows"
        assert result.severity == "low"
        assert "optional_field" in result.message
        assert "MCAR" in result.message
        assert result.details["reasoning"]
    
    # Integration Tests
    
    def test_missing_rule_priorities(self):
        """Test that missing rules have correct priority ordering"""
        high = HighMissingRule()
        moderate = ModerateMissingRule()
        pattern = PatternMissingRule()
        random = RandomMissingRule()
        
        assert high.priority > moderate.priority > pattern.priority > random.priority
    
    def test_multiple_missing_rules_same_column(self, base_context, column_template):
        """Test that a column can trigger multiple missing rules"""
        column_template["type"] = "numeric"
        column_template["signals"]["high_missing"] = True
        column_template["signals"]["moderate_missing"] = False
        column_template["missing"]["percent"] = 60
        column_template["missing_pattern"]["transitions"] = 2
        column_template["missing_pattern"]["max_consecutive_missing"] = 200
        base_context["columns"] = [column_template]
        
        high_rule = HighMissingRule()
        pattern_rule = PatternMissingRule()
        
        # Both should apply
        assert high_rule.applies(base_context) is True
        assert pattern_rule.applies(base_context) is True


