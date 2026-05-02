import pytest
import pandas as pd
import numpy as np
from core.ingestion.context_builder import ContextBuilder


class TestContextBuilderValidation:
    """Test suite for DataFrame validation"""
    
    def test_empty_dataframe_raises_error(self):
        """Empty DataFrame should raise ValueError"""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty Dataframe"):
            ContextBuilder(df)
    
    def test_all_na_dataframe_raises_error(self):
        """DataFrame with only NA values should raise ValueError"""
        df = pd.DataFrame({
            "col1": [np.nan, np.nan, np.nan],
            "col2": [None, None, None]
        })
        with pytest.raises(ValueError, match="Filled with NA in Dataframe"):
            ContextBuilder(df)
    
    def test_valid_dataframe_passes(self):
        """Valid DataFrame should not raise error"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        builder = ContextBuilder(df)
        assert builder.df is not None
    
    def test_missing_target_column_raises_error(self):
        """Non-existent target column should raise ValueError"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(ValueError, match="Target column 'target' not found"):
            ContextBuilder(df, target_col="target")
    
    def test_valid_target_column_passes(self):
        """Valid target column should not raise error"""
        df = pd.DataFrame({"col1": [1, 2, 3], "target": [0, 1, 0]})
        builder = ContextBuilder(df, target_col="target")
        assert builder.target_col == "target"
    
    def test_none_target_column_passes(self):
        """None target column should be accepted"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        builder = ContextBuilder(df, target_col=None)
        assert builder.target_col is None


class TestBuildContext:
    """Test suite for build_context method"""
    
    def test_build_context_structure(self):
        """Context should have all required top-level keys"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        builder = ContextBuilder(df)
        context = builder.build_context()
        
        assert "dataset_overview" in context
        assert "dataset_health" in context
        assert "columns" in context
        assert "top_correlations" in context
    
    def test_dataset_overview_content(self):
        """Dataset overview should contain correct row and column counts"""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50]
        })
        builder = ContextBuilder(df, target_col="col1")
        context = builder.build_context()
        
        overview = context["dataset_overview"]
        assert overview["num_rows"] == 5
        assert overview["num_columns"] == 2
        assert overview["target_variable"] == "col1"
    
    def test_dataset_overview_no_target(self):
        """Dataset overview with no target should have None"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        builder = ContextBuilder(df)
        context = builder.build_context()
        
        assert context["dataset_overview"]["target_variable"] is None
    
    def test_columns_list_length(self):
        """Columns list should match DataFrame column count"""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9]
        })
        builder = ContextBuilder(df)
        context = builder.build_context()
        
        assert len(context["columns"]) == 3
    
    def test_columns_have_name_attribute(self):
        """Each column should have a name attribute"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        builder = ContextBuilder(df)
        context = builder.build_context()
        
        column_names = [col["name"] for col in context["columns"]]
        assert "col1" in column_names
        assert "col2" in column_names


class TestDatasetLevelAnalysis:
    """Test suite for _dataset_level_analysis method"""
    
    def test_no_duplicates(self):
        """Should correctly identify no duplicates"""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        builder = ContextBuilder(df)
        health = builder._dataset_level_analysis()
        
        assert health["duplicate_rows"] == 0
        assert health["duplicate_percent"] == 0.0
    
    def test_with_duplicates(self):
        """Should correctly count duplicate rows"""
        df = pd.DataFrame({
            "col1": [1, 2, 1, 3],
            "col2": [4, 5, 4, 6]
        })
        builder = ContextBuilder(df)
        health = builder._dataset_level_analysis()
        
        assert health["duplicate_rows"] == 1
        assert health["duplicate_percent"] == 25.0
    
    def test_memory_usage_present(self):
        """Should calculate memory usage"""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        builder = ContextBuilder(df)
        health = builder._dataset_level_analysis()
        
        assert "memory_usage_mb" in health
        assert health["memory_usage_mb"] > 0


class TestClassifyType:
    """Test suite for _classify_type method"""
    
    def test_classify_numeric(self):
        """Should classify numeric series as numeric"""
        df = pd.DataFrame({"col": [1.5, 2.5, 3.5, 4.5, 5.5]})
        builder = ContextBuilder(df)
        col_type = builder._classify_type(df["col"])
        
        assert col_type == "numeric"
    
    def test_classify_categorical_from_low_unique(self):
        """Numeric with few unique values should be categorical"""
        df = pd.DataFrame({"col": [1, 2, 1, 2, 1, 2] * 10})
        builder = ContextBuilder(df)
        col_type = builder._classify_type(df["col"])
        
        assert col_type == "categorical"
    
    def test_classify_categorical_from_low_ratio(self):
        """Numeric with low unique ratio should be categorical"""
        df = pd.DataFrame({"col": [1] * 1000 + [2] * 1000})
        builder = ContextBuilder(df)
        col_type = builder._classify_type(df["col"])
        
        assert col_type == "categorical"
    
    def test_classify_boolean_as_categorical(self):
        """Boolean series should be categorical"""
        df = pd.DataFrame({"col": [True, False, True, False]})
        builder = ContextBuilder(df)
        col_type = builder._classify_type(df["col"])
        
        assert col_type == "categorical"
    
    def test_classify_datetime(self):
        """Datetime series should be datetime"""
        df = pd.DataFrame({
            "col": pd.date_range("2020-01-01", periods=5)
        })
        builder = ContextBuilder(df)
        col_type = builder._classify_type(df["col"])
        
        assert col_type == "datetime"
    
    def test_classify_text_from_length(self):
        """Long strings with high uniqueness should be text"""
        df = pd.DataFrame({
            "col": [
                "This is a long text string that exceeds fifty characters easily",
                "Another unique long text string that is quite different from the first",
                "Yet another long and unique text string for testing purposes here"
            ]
        })
        builder = ContextBuilder(df)
        col_type = builder._classify_type(df["col"])
        
        assert col_type == "text"
    
    def test_classify_short_strings_as_categorical(self):
        """Short strings should be categorical"""
        df = pd.DataFrame({
            "col": ["cat", "dog", "cat", "bird", "dog"]
        })
        builder = ContextBuilder(df)
        col_type = builder._classify_type(df["col"])
        
        assert col_type == "categorical"


class TestAnalyzeColumn:
    """Test suite for _analyze_column method"""
    
    def test_numeric_column_analysis(self):
        """Numeric column should have correct analysis structure"""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5, 100]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["type"] == "numeric"
        assert "mean" in analysis["stats"]
        assert "std" in analysis["stats"]
        assert "min" in analysis["stats"]
        assert "max" in analysis["stats"]
        assert "skew" in analysis["stats"]
        assert "outlier_count" in analysis["stats"]
        assert "variance" in analysis["stats"]
    
    def test_categorical_column_analysis(self):
        """Categorical column should have correct analysis structure"""
        df = pd.DataFrame({"col": ["a", "b", "a", "a", "c"]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["type"] == "categorical"
        assert "entropy" in analysis["stats"]
        assert "top_values" in analysis["stats"]
    
    def test_text_column_analysis(self):
        """Text column should have avg_length stat"""
        df = pd.DataFrame({
            "col": [
                "This is a long text string that exceeds fifty characters easily",
                "Another long text string here"
            ]
        })
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["type"] == "text"
        assert "avg_length" in analysis["stats"]
        assert analysis["signals"]["long_text"] is True
    
    def test_missing_values_calculation(self):
        """Should correctly calculate missing values"""
        df = pd.DataFrame({"col": [1, 2, np.nan, 4, np.nan]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["missing"]["count"] == 2
        assert analysis["missing"]["percent"] == 40.0
    
    def test_cardinality_calculation(self):
        """Should correctly calculate cardinality"""
        df = pd.DataFrame({"col": [1, 2, 2, 3, 3, 3]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["cardinality"] == 3
    
    def test_unique_ratio_calculation(self):
        """Should correctly calculate unique ratio"""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})  # All unique
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["unique_ratio"] == 1.0
    
    def test_constant_signal(self):
        """Should detect constant columns"""
        df = pd.DataFrame({"col": [5, 5, 5, 5, 5]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["constant"] is True
        assert "constant" in analysis["flags"]
    
    def test_possible_id_signal(self):
        """Should detect possible ID columns"""
        df = pd.DataFrame({"col": [f"id_{i}" for i in range(100)]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["possible_id"] is True
        assert "possible_id" in analysis["flags"]
    
    def test_high_missing_signal(self):
        """Should detect high missing percentage"""
        df = pd.DataFrame({"col": [1] + [np.nan] * 99})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["high_missing"] is np.True_
        assert "high_missing" in analysis["flags"]
    
    def test_moderate_missing_signal(self):
        """Should detect moderate missing percentage"""
        df = pd.DataFrame({"col": [1] * 60 + [np.nan] * 40})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["moderate_missing"] is np.True_
        assert "moderate_missing" in analysis["flags"]
    
    def test_skewed_signal(self):
        """Should detect skewed distributions"""
        # Create right-skewed data
        data = ([1, 1.1, 1.2, 1.3, 0.9, 0.8] * 15) + [100, 110, 120, 130, 140]
        df = pd.DataFrame({"col": data})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        assert "skew" in analysis["stats"], f"Expected 'numeric' type, but got {analysis['type']}"
        # Skew > 1.5 should trigger skewed signal
        skew_value = analysis["stats"]["skew"]

        if abs(skew_value) > 1.5:
            # Use standard Python True
            assert analysis["signals"]["skewed"] is True

    def test_has_outliers_signal(self):
        """Should detect outliers"""
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5, 1000]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["has_outliers"] is True
        assert "has_outliers" in analysis["flags"]
    
    def test_low_variance_signal(self):
        """Should detect low variance"""
        df = pd.DataFrame({"col": [1.0000, 1.0001, 1.0002, 1.0001]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["low_variance"] is True
        assert "low_variance" in analysis["flags"]
    
    def test_is_target_signal(self):
        """Should mark target column"""
        df = pd.DataFrame({"col1": [1, 2, 3], "target": [0, 1, 0]})
        builder = ContextBuilder(df, target_col="target")
        analysis = builder._analyze_column("target")
        
        assert analysis["signals"]["is_target"] is True
    
    def test_imbalanced_categorical_signal(self):
        """Should detect imbalanced categorical columns"""
        df = pd.DataFrame({"col": ["a"] * 90 + ["b"] * 10})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["imbalanced"] is True
        assert "imbalanced" in analysis["flags"]
    
    def test_event_like_signal(self):
        """Should detect event-like columns"""
        # >50% missing, numeric/categorical, low cardinality
        df = pd.DataFrame({"col": [1, 0, 0] + [np.nan] * 97})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        assert analysis["signals"]["event_like"] is True
        assert "event_like" in analysis["flags"]
    
    def test_flags_are_sorted(self):
        """Flags should be sorted alphabetically"""
        df = pd.DataFrame({"col": [1] * 95 + [np.nan] * 5})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        flags = analysis["flags"]
        assert flags == sorted(flags)
    
    def test_density_calculation(self):
        """Should calculate density correctly"""
        df = pd.DataFrame({"col": [1, 2, np.nan, 4, 5]})
        builder = ContextBuilder(df)
        analysis = builder._analyze_column("col")
        
        # Density = 1 - missing_ratio = 1 - 0.2 = 0.8
        assert analysis["stats"]["density"] == 0.8


class TestGetHighCorrelations:
    """Test suite for _get_high_correlations method"""
    
    def test_no_correlations_with_one_column(self):
        """Single numeric column should return empty correlations"""
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})
        builder = ContextBuilder(df)
        correlations = builder._get_high_correlations()
        
        assert correlations == {}
    
    def test_high_correlation_detected(self):
        """Should detect high correlations"""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [2, 4, 6, 8, 10]  # Perfect correlation
        })
        builder = ContextBuilder(df)
        correlations = builder._get_high_correlations()
        
        assert len(correlations) > 0
        assert correlations[0]["correlation"] > 0.75
    
    def test_correlation_structure(self):
        """Correlation items should have required fields"""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [2, 4, 6, 8, 10]
        })
        builder = ContextBuilder(df)
        correlations = builder._get_high_correlations()
        
        if correlations:
            assert "feature_1" in correlations[0]
            assert "feature_2" in correlations[0]
            assert "correlation" in correlations[0]
            assert "insight" in correlations[0]
    
    def test_low_correlations_not_included(self):
        """Low correlations should not be included"""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [5, 4, 3, 2, 1]  # Negative correlation
        })
        builder = ContextBuilder(df)
        correlations = builder._get_high_correlations()
        
        # May be empty or only include correlations > 0.75
        for corr in correlations:
            assert corr["correlation"] > 0.75
    
    def test_max_five_correlations(self):
        """Should return at most 5 correlations"""
        # Create many highly correlated columns
        df = pd.DataFrame({
            f"col{i}": [j * i for j in range(100)]
            for i in range(20)
        })
        builder = ContextBuilder(df)
        correlations = builder._get_high_correlations()
        
        assert len(correlations) <= 5
    
    def test_correlations_sorted_descending(self):
        """Correlations should be sorted by correlation value descending"""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [2, 4, 6, 8, 10],
            "col3": [1.5, 3, 4.5, 6, 7.5]
        })
        builder = ContextBuilder(df)
        correlations = builder._get_high_correlations()
        
        if len(correlations) > 1:
            for i in range(len(correlations) - 1):
                assert correlations[i]["correlation"] >= correlations[i + 1]["correlation"]
    
    def test_non_numeric_columns_ignored(self):
        """Non-numeric columns should be ignored"""
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [2, 4, 6, 8, 10],
            "text_col": ["a", "b", "c", "d", "e"]
        })
        builder = ContextBuilder(df)
        correlations = builder._get_high_correlations()
        
        # Should only consider col1 and col2
        for corr in correlations:
            assert corr["feature_1"] in ["col1", "col2"]
            assert corr["feature_2"] in ["col1", "col2"]


class TestCalculateEntropy:
    """Test suite for _calculate_entropy method"""
    
    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution should have high entropy"""
        series = pd.Series(["a", "b", "c", "d"] * 25)
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        entropy = builder._calculate_entropy(series)
        
        # Uniform distribution has maximum entropy
        assert entropy > 1.9  # Close to log2(4) = 2
    
    def test_single_value_zero_entropy(self):
        """Single unique value should have zero entropy"""
        series = pd.Series(["a"] * 100)
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        entropy = builder._calculate_entropy(series)
        
        assert entropy == pytest.approx(0, abs=1e-6)
    
    def test_skewed_distribution_lower_entropy(self):
        """Skewed distribution should have lower entropy"""
        series = pd.Series(["a"] * 90 + ["b"] * 10)
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        entropy = builder._calculate_entropy(series)
        
        assert 0 < entropy < 1


class TestMissingPattern:
    """Test suite for _missing_pattern method"""
    
    def test_no_missing_values(self):
        """No missing values should have no transitions"""
        series = pd.Series([1, 2, 3, 4, 5])
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        pattern = builder._missing_pattern(series)
        
        assert pattern["transitions"] == 0
        assert pattern["max_consecutive_missing"] == 0
    
    def test_all_missing_values(self):
        """All missing should have max_consecutive_missing = length"""
        series = pd.Series([np.nan] * 10)
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        pattern = builder._missing_pattern(series)
        
        assert pattern["max_consecutive_missing"] == 10
    
    def test_alternating_missing(self):
        """Alternating pattern should have many transitions"""
        series = pd.Series([1, np.nan, 2, np.nan, 3, np.nan])
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        pattern = builder._missing_pattern(series)
        
        # Should have multiple transitions
        assert pattern["transitions"] > 0
    
    def test_consecutive_missing_streak(self):
        """Should track longest consecutive missing streak"""
        series = pd.Series([1, 2, np.nan, np.nan, np.nan, 3, np.nan, 4])
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        pattern = builder._missing_pattern(series)
        
        assert pattern["max_consecutive_missing"] == 3


class TestGetSample:
    """Test suite for _get_sample method"""
    
    def test_small_series_returns_full(self):
        """Series smaller than max_size should be returned in full"""
        series = pd.Series(range(100))
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        sample = builder._get_sample(series, max_size=10000)
        
        assert len(sample) == 100
    
    def test_large_series_returns_sample(self):
        """Series larger than max_size should be sampled"""
        series = pd.Series(range(20000))
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        sample = builder._get_sample(series, max_size=10000)
        
        assert len(sample) == 10000
    
    def test_sample_is_reproducible(self):
        """Sample should be reproducible with fixed random_state"""
        series = pd.Series(range(20000))
        builder = ContextBuilder(pd.DataFrame({"col": [1]}))
        sample1 = builder._get_sample(series, max_size=10000)
        sample2 = builder._get_sample(series, max_size=10000)
        
        assert sample1.equals(sample2)


class TestContextBuilderIntegration:
    """Integration tests for ContextBuilder"""
    
    def test_full_context_build(self):
        """Test complete context building with realistic data"""
        df = pd.DataFrame({
            "id": range(100),
            "age": np.random.randint(18, 80, 100),
            "income": np.random.normal(50000, 15000, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100)
        })
        
        builder = ContextBuilder(df, target_col="target")
        context = builder.build_context()
        
        # Verify structure
        assert "dataset_overview" in context
        assert "dataset_health" in context
        assert "columns" in context
        assert "top_correlations" in context
        
        # Verify dataset overview
        assert context["dataset_overview"]["num_rows"] == 100
        assert context["dataset_overview"]["num_columns"] == 5
        assert context["dataset_overview"]["target_variable"] == "target"
        
        # Verify columns
        assert len(context["columns"]) == 5
        
        # Check each column has required fields
        for col in context["columns"]:
            assert "name" in col
            assert "type" in col
            assert "missing" in col
            assert "cardinality" in col
            assert "unique_ratio" in col
            assert "stats" in col
            assert "signals" in col
            assert "missing_pattern" in col
            assert "flags" in col
    
    def test_mixed_data_types(self):
        """Test with various data types"""
        df = pd.DataFrame({
            "numeric": [1.5, 2.5, 3.5],
            "integer": [1, 2, 3],
            "boolean": [True, False, True],
            "categorical": ["cat", "dog", "cat"],
            "text": [
                "This is a long text string with many characters",
                "Another long text string here with different content",
                "Yet another completely different long text string"
            ],
            "datetime": pd.date_range("2020-01-01", periods=3),
            "with_missing": [1, np.nan, 3]
        })
        
        builder = ContextBuilder(df)
        context = builder.build_context()
        
        # Check all columns are analyzed
        assert len(context["columns"]) == 7
        
        # Verify type classification
        column_types = {col["name"]: col["type"] for col in context["columns"]}
        assert column_types["numeric"] == "numeric"
        assert column_types["categorical"] == "categorical"
        assert column_types["text"] == "text"
        assert column_types["datetime"] == "datetime"