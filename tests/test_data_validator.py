import pytest
from pydantic import ValidationError
from core.ingestion.data_validator import (
    MissingInfo,
    MissingPattern,
    ColumnStats,
    ColumnSignals,
    ColumnAnalysis,
    DatasetOverview,
    DatasetHealth,
    Correlation,
    CanonicalData,
    Metadata,
    CanonicalizedOutput
)


class TestMissingInfo:
    """Test suite for MissingInfo model"""
    
    def test_valid_missing_info(self):
        """Valid MissingInfo should be created successfully"""
        missing = MissingInfo(count=10, percent=5.5)
        assert missing.count == 10
        assert missing.percent == 5.5
    
    def test_missing_info_zero_values(self):
        """Zero values should be valid"""
        missing = MissingInfo(count=0, percent=0.0)
        assert missing.count == 0
        assert missing.percent == 0.0
    
    def test_missing_required_fields(self):
        """Missing required fields should raise ValidationError"""
        with pytest.raises(ValidationError):
            MissingInfo(count=10)
        
        with pytest.raises(ValidationError):
            MissingInfo(percent=5.5)


class TestMissingPattern:
    """Test suite for MissingPattern model"""
    
    def test_valid_missing_pattern(self):
        """Valid MissingPattern should be created successfully"""
        pattern = MissingPattern(transitions=5, max_consecutive_missing=10)
        assert pattern.transitions == 5
        assert pattern.max_consecutive_missing == 10
    
    def test_missing_pattern_zero_values(self):
        """Zero values should be valid"""
        pattern = MissingPattern(transitions=0, max_consecutive_missing=0)
        assert pattern.transitions == 0
        assert pattern.max_consecutive_missing == 0


class TestColumnStats:
    """Test suite for ColumnStats model"""
    
    def test_minimal_column_stats(self):
        """Only density is required"""
        stats = ColumnStats(density=0.95)
        assert stats.density == 0.95
        assert stats.entropy is None
        assert stats.mean is None
    
    def test_full_numeric_stats(self):
        """Complete numeric stats should be valid"""
        stats = ColumnStats(
            density=0.95,
            mean=50.5,
            std=15.3,
            min=10.0,
            max=100.0,
            skew=0.5,
            outlier_count=5,
            variance=234.09
        )
        assert stats.mean == 50.5
        assert stats.std == 15.3
        assert stats.variance == 234.09
    
    def test_categorical_stats(self):
        """Categorical stats should be valid"""
        stats = ColumnStats(
            density=1.0,
            entropy=2.5,
            top_values=["cat", "dog", "bird"]
        )
        assert stats.entropy == 2.5
        assert stats.top_values == ["cat", "dog", "bird"]
    
    def test_text_stats(self):
        """Text stats should be valid"""
        stats = ColumnStats(
            density=0.99,
            avg_length=125.5
        )
        assert stats.avg_length == 125.5
    
    def test_top_values_with_mixed_types(self):
        """top_values can contain mixed types"""
        stats = ColumnStats(
            density=1.0,
            top_values=["string", 123, True]
        )
        assert stats.top_values == ["string", 123, True]


class TestColumnSignals:
    """Test suite for ColumnSignals model"""
    
    def test_minimal_column_signals(self):
        """All required signals must be present"""
        signals = ColumnSignals(
            constant=False,
            possible_id=False,
            high_missing=False,
            moderate_missing=False,
            high_cardinality=False,
            event_like=False,
            unique_ratio=0.5
        )
        assert signals.constant is False
        assert signals.unique_ratio == 0.5
    
    def test_optional_signals(self):
        """Optional signals can be None or set"""
        signals = ColumnSignals(
            constant=True,
            possible_id=False,
            high_missing=False,
            moderate_missing=False,
            high_cardinality=True,
            event_like=False,
            unique_ratio=0.95,
            imbalanced=True,
            skewed=True,
            has_outliers=True
        )
        assert signals.imbalanced is True
        assert signals.skewed is True
        assert signals.has_outliers is True
    
    def test_numeric_signals(self):
        """Numeric-specific signals"""
        signals = ColumnSignals(
            constant=False,
            possible_id=False,
            high_missing=False,
            moderate_missing=False,
            high_cardinality=False,
            event_like=False,
            unique_ratio=1.0,
            skewed=True,
            has_outliers=True,
            low_variance=False
        )
        assert signals.skewed is True
        assert signals.has_outliers is True
        assert signals.low_variance is False
    
    def test_text_signals(self):
        """Text-specific signals"""
        signals = ColumnSignals(
            constant=False,
            possible_id=False,
            high_missing=False,
            moderate_missing=False,
            high_cardinality=True,
            event_like=False,
            unique_ratio=0.99,
            long_text=True
        )
        assert signals.long_text is True


class TestColumnAnalysis:
    """Test suite for ColumnAnalysis model"""
    
    def test_valid_column_analysis(self):
        """Complete valid column analysis"""
        analysis = ColumnAnalysis(
            name="age",
            type="numeric",
            missing=MissingInfo(count=5, percent=5.0),
            cardinality=50,
            unique_ratio=0.5,
            stats=ColumnStats(
                density=0.95,
                mean=35.5,
                std=12.3,
                min=18.0,
                max=80.0,
                skew=0.5,
                outlier_count=3,
                variance=151.29
            ),
            signals=ColumnSignals(
                constant=False,
                possible_id=False,
                high_missing=False,
                moderate_missing=False,
                high_cardinality=False,
                event_like=False,
                unique_ratio=0.5,
                skewed=False,
                has_outliers=True,
                low_variance=False
            ),
            missing_pattern=MissingPattern(transitions=2, max_consecutive_missing=3),
            flags=["has_outliers"]
        )
        assert analysis.name == "age"
        assert analysis.type == "numeric"
        assert analysis.cardinality == 50
    
    def test_flags_must_be_sorted(self):
        """Flags not sorted should raise ValidationError"""
        with pytest.raises(ValidationError, match="sorted alphabetically"):
            ColumnAnalysis(
                name="test",
                type="numeric",
                missing=MissingInfo(count=0, percent=0.0),
                cardinality=10,
                unique_ratio=1.0,
                stats=ColumnStats(density=1.0),
                signals=ColumnSignals(
                    constant=False,
                    possible_id=False,
                    high_missing=False,
                    moderate_missing=False,
                    high_cardinality=True,
                    event_like=False,
                    unique_ratio=1.0
                ),
                missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                flags=["skewed", "has_outliers", "constant"]  # Not sorted
            )
    
    def test_flags_sorted_valid(self):
        """Properly sorted flags should be valid"""
        analysis = ColumnAnalysis(
            name="test",
            type="numeric",
            missing=MissingInfo(count=0, percent=0.0),
            cardinality=10,
            unique_ratio=1.0,
            stats=ColumnStats(density=1.0),
            signals=ColumnSignals(
                constant=False,
                possible_id=False,
                high_missing=False,
                moderate_missing=False,
                high_cardinality=True,
                event_like=False,
                unique_ratio=1.0
            ),
            missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
            flags=["constant", "has_outliers", "skewed"]  # Sorted
        )
        assert analysis.flags == ["constant", "has_outliers", "skewed"]
    
    def test_empty_flags(self):
        """Empty flags list should be valid"""
        analysis = ColumnAnalysis(
            name="test",
            type="numeric",
            missing=MissingInfo(count=0, percent=0.0),
            cardinality=10,
            unique_ratio=1.0,
            stats=ColumnStats(density=1.0),
            signals=ColumnSignals(
                constant=False,
                possible_id=False,
                high_missing=False,
                moderate_missing=False,
                high_cardinality=False,
                event_like=False,
                unique_ratio=1.0
            ),
            missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
            flags=[]
        )
        assert analysis.flags == []


class TestDatasetOverview:
    """Test suite for DatasetOverview model"""
    
    def test_valid_dataset_overview(self):
        """Valid dataset overview"""
        overview = DatasetOverview(num_rows=1000, num_columns=10)
        assert overview.num_rows == 1000
        assert overview.num_columns == 10
    
    def test_zero_rows(self):
        """Zero rows should be valid (though unusual)"""
        overview = DatasetOverview(num_rows=0, num_columns=5)
        assert overview.num_rows == 0


class TestDatasetHealth:
    """Test suite for DatasetHealth model"""
    
    def test_valid_dataset_health(self):
        """Valid dataset health metrics"""
        health = DatasetHealth(
            duplicate_rows=10,
            duplicate_percent=5.5,
            memory_usage_mb=2.5
        )
        assert health.duplicate_rows == 10
        assert health.duplicate_percent == 5.5
        assert health.memory_usage_mb == 2.5
    
    def test_no_duplicates(self):
        """No duplicates should be valid"""
        health = DatasetHealth(
            duplicate_rows=0,
            duplicate_percent=0.0,
            memory_usage_mb=1.0
        )
        assert health.duplicate_rows == 0


class TestCorrelation:
    """Test suite for Correlation model"""
    
    def test_valid_correlation(self):
        """Valid correlation pair"""
        corr = Correlation(
            feature_1="age",
            feature_2="income",
            correlation=0.85,
            insight="potential multicollinearity"
        )
        assert corr.feature_1 == "age"
        assert corr.feature_2 == "income"
        assert corr.correlation == 0.85
        assert corr.insight == "potential multicollinearity"
    
    def test_perfect_correlation(self):
        """Perfect correlation (1.0) should be valid"""
        corr = Correlation(
            feature_1="col1",
            feature_2="col2",
            correlation=1.0,
            insight="perfect correlation"
        )
        assert corr.correlation == 1.0
    
    def test_negative_correlation(self):
        """Correlation can be any float"""
        corr = Correlation(
            feature_1="col1",
            feature_2="col2",
            correlation=-0.95,
            insight="strong negative correlation"
        )
        assert corr.correlation == -0.95


class TestCanonicalData:
    """Test suite for CanonicalData model"""
    
    def test_valid_canonical_data(self):
        """Valid canonical data structure"""
        data = CanonicalData(
            columns=[
                ColumnAnalysis(
                    name="age",
                    type="numeric",
                    missing=MissingInfo(count=0, percent=0.0),
                    cardinality=50,
                    unique_ratio=0.5,
                    stats=ColumnStats(density=1.0),
                    signals=ColumnSignals(
                        constant=False,
                        possible_id=False,
                        high_missing=False,
                        moderate_missing=False,
                        high_cardinality=False,
                        event_like=False,
                        unique_ratio=0.5
                    ),
                    missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                    flags=[]
                )
            ],
            dataset_health=DatasetHealth(
                duplicate_rows=0,
                duplicate_percent=0.0,
                memory_usage_mb=1.5
            ),
            dataset_overview=DatasetOverview(num_rows=100, num_columns=1),
            top_correlations=[]
        )
        assert len(data.columns) == 1
        assert data.dataset_overview.num_rows == 100
    
    def test_columns_must_be_sorted(self):
        """Columns not sorted by type then name should raise ValidationError"""
        with pytest.raises(ValidationError, match="sorted by type, then by name"):
            CanonicalData(
                columns=[
                    ColumnAnalysis(
                        name="z_col",
                        type="numeric",
                        missing=MissingInfo(count=0, percent=0.0),
                        cardinality=10,
                        unique_ratio=1.0,
                        stats=ColumnStats(density=1.0),
                        signals=ColumnSignals(
                            constant=False,
                            possible_id=False,
                            high_missing=False,
                            moderate_missing=False,
                            high_cardinality=False,
                            event_like=False,
                            unique_ratio=1.0
                        ),
                        missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                        flags=[]
                    ),
                    ColumnAnalysis(
                        name="a_col",
                        type="categorical",
                        missing=MissingInfo(count=0, percent=0.0),
                        cardinality=5,
                        unique_ratio=0.5,
                        stats=ColumnStats(density=1.0),
                        signals=ColumnSignals(
                            constant=False,
                            possible_id=False,
                            high_missing=False,
                            moderate_missing=False,
                            high_cardinality=False,
                            event_like=False,
                            unique_ratio=0.5
                        ),
                        missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                        flags=[]
                    )
                ],
                dataset_health=DatasetHealth(
                    duplicate_rows=0,
                    duplicate_percent=0.0,
                    memory_usage_mb=1.0
                ),
                dataset_overview=DatasetOverview(num_rows=10, num_columns=2),
                top_correlations=[]
            )
    
    def test_columns_sorted_correctly(self):
        """Properly sorted columns should be valid"""
        data = CanonicalData(
            columns=[
                ColumnAnalysis(
                    name="a_col",
                    type="categorical",
                    missing=MissingInfo(count=0, percent=0.0),
                    cardinality=5,
                    unique_ratio=0.5,
                    stats=ColumnStats(density=1.0),
                    signals=ColumnSignals(
                        constant=False,
                        possible_id=False,
                        high_missing=False,
                        moderate_missing=False,
                        high_cardinality=False,
                        event_like=False,
                        unique_ratio=0.5
                    ),
                    missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                    flags=[]
                ),
                ColumnAnalysis(
                    name="b_col",
                    type="numeric",
                    missing=MissingInfo(count=0, percent=0.0),
                    cardinality=10,
                    unique_ratio=1.0,
                    stats=ColumnStats(density=1.0),
                    signals=ColumnSignals(
                        constant=False,
                        possible_id=False,
                        high_missing=False,
                        moderate_missing=False,
                        high_cardinality=False,
                        event_like=False,
                        unique_ratio=1.0
                    ),
                    missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                    flags=[]
                ),
                ColumnAnalysis(
                    name="z_col",
                    type="numeric",
                    missing=MissingInfo(count=0, percent=0.0),
                    cardinality=10,
                    unique_ratio=1.0,
                    stats=ColumnStats(density=1.0),
                    signals=ColumnSignals(
                        constant=False,
                        possible_id=False,
                        high_missing=False,
                        moderate_missing=False,
                        high_cardinality=False,
                        event_like=False,
                        unique_ratio=1.0
                    ),
                    missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                    flags=[]
                )
            ],
            dataset_health=DatasetHealth(
                duplicate_rows=0,
                duplicate_percent=0.0,
                memory_usage_mb=1.0
            ),
            dataset_overview=DatasetOverview(num_rows=10, num_columns=3),
            top_correlations=[]
        )
        # Should not raise an error
        assert len(data.columns) == 3
    
    def test_empty_columns_list(self):
        """Empty columns list should be valid"""
        data = CanonicalData(
            columns=[],
            dataset_health=DatasetHealth(
                duplicate_rows=0,
                duplicate_percent=0.0,
                memory_usage_mb=0.1
            ),
            dataset_overview=DatasetOverview(num_rows=0, num_columns=0),
            top_correlations=[]
        )
        assert len(data.columns) == 0
    
    def test_with_correlations(self):
        """CanonicalData with correlations"""
        data = CanonicalData(
            columns=[],
            dataset_health=DatasetHealth(
                duplicate_rows=0,
                duplicate_percent=0.0,
                memory_usage_mb=1.0
            ),
            dataset_overview=DatasetOverview(num_rows=100, num_columns=0),
            top_correlations=[
                Correlation(
                    feature_1="col1",
                    feature_2="col2",
                    correlation=0.95,
                    insight="potential multicollinearity"
                )
            ]
        )
        assert len(data.top_correlations) == 1


class TestMetadata:
    """Test suite for Metadata model"""
    
    def test_valid_metadata(self):
        """Valid metadata structure"""
        metadata = Metadata(
            state_uuid="a" * 32,
            fingerprint="b" * 64,
            schema_version="1.0.0"
        )
        assert metadata.state_uuid == "a" * 32
        assert metadata.fingerprint == "b" * 64
        assert metadata.schema_version == "1.0.0"
    
    def test_metadata_with_various_versions(self):
        """Different schema versions should be valid"""
        metadata = Metadata(
            state_uuid="x" * 32,
            fingerprint="y" * 64,
            schema_version="2.5.1"
        )
        assert metadata.schema_version == "2.5.1"


class TestCanonicalizedOutput:
    """Test suite for CanonicalizedOutput model"""
    
    def test_valid_canonicalized_output(self):
        """Complete valid canonicalized output"""
        output = CanonicalizedOutput(
            metadata=Metadata(
                state_uuid="a" * 32,
                fingerprint="b" * 64,
                schema_version="1.0.0"
            ),
            canonical_data=CanonicalData(
                columns=[
                    ColumnAnalysis(
                        name="age",
                        type="numeric",
                        missing=MissingInfo(count=0, percent=0.0),
                        cardinality=50,
                        unique_ratio=0.5,
                        stats=ColumnStats(density=1.0, mean=35.0),
                        signals=ColumnSignals(
                            constant=False,
                            possible_id=False,
                            high_missing=False,
                            moderate_missing=False,
                            high_cardinality=False,
                            event_like=False,
                            unique_ratio=0.5
                        ),
                        missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                        flags=[]
                    )
                ],
                dataset_health=DatasetHealth(
                    duplicate_rows=0,
                    duplicate_percent=0.0,
                    memory_usage_mb=1.5
                ),
                dataset_overview=DatasetOverview(num_rows=100, num_columns=1),
                top_correlations=[]
            )
        )
        assert output.metadata.schema_version == "1.0.0"
        assert len(output.canonical_data.columns) == 1
    
    def test_complete_realistic_output(self):
        """Test with realistic complete structure"""
        output = CanonicalizedOutput(
            metadata=Metadata(
                state_uuid="1234567890abcdef1234567890abcdef",
                fingerprint="fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
                schema_version="1.0.0"
            ),
            canonical_data=CanonicalData(
                columns=[
                    ColumnAnalysis(
                        name="category",
                        type="categorical",
                        missing=MissingInfo(count=5, percent=5.0),
                        cardinality=3,
                        unique_ratio=0.03,
                        stats=ColumnStats(density=0.95, entropy=1.5, top_values=["A", "B", "C"]),
                        signals=ColumnSignals(
                            constant=False,
                            possible_id=False,
                            high_missing=False,
                            moderate_missing=False,
                            high_cardinality=False,
                            event_like=False,
                            unique_ratio=0.03,
                            imbalanced=True
                        ),
                        missing_pattern=MissingPattern(transitions=3, max_consecutive_missing=2),
                        flags=["imbalanced"]
                    ),
                    ColumnAnalysis(
                        name="age",
                        type="numeric",
                        missing=MissingInfo(count=0, percent=0.0),
                        cardinality=50,
                        unique_ratio=0.5,
                        stats=ColumnStats(
                            density=1.0,
                            mean=35.5,
                            std=12.3,
                            min=18.0,
                            max=80.0,
                            skew=0.5,
                            outlier_count=3,
                            variance=151.29
                        ),
                        signals=ColumnSignals(
                            constant=False,
                            possible_id=False,
                            high_missing=False,
                            moderate_missing=False,
                            high_cardinality=False,
                            event_like=False,
                            unique_ratio=0.5,
                            skewed=False,
                            has_outliers=True,
                            low_variance=False
                        ),
                        missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
                        flags=["has_outliers"]
                    )
                ],
                dataset_health=DatasetHealth(
                    duplicate_rows=5,
                    duplicate_percent=5.0,
                    memory_usage_mb=2.5
                ),
                dataset_overview=DatasetOverview(num_rows=100, num_columns=2),
                top_correlations=[
                    Correlation(
                        feature_1="age",
                        feature_2="income",
                        correlation=0.85,
                        insight="potential multicollinearity"
                    )
                ]
            )
        )
        
        # Validate structure
        assert output.metadata.state_uuid == "1234567890abcdef1234567890abcdef"
        assert output.canonical_data.dataset_overview.num_rows == 100
        assert len(output.canonical_data.columns) == 2
        assert len(output.canonical_data.top_correlations) == 1
        
        # Validate sorting
        assert output.canonical_data.columns[0].type == "categorical"
        assert output.canonical_data.columns[1].type == "numeric"


class TestValidationEdgeCases:
    """Test edge cases and validation scenarios"""
    
    def test_large_cardinality(self):
        """Very large cardinality should be valid"""
        analysis = ColumnAnalysis(
            name="id",
            type="text",
            missing=MissingInfo(count=0, percent=0.0),
            cardinality=1000000,
            unique_ratio=1.0,
            stats=ColumnStats(density=1.0),
            signals=ColumnSignals(
                constant=False,
                possible_id=True,
                high_missing=False,
                moderate_missing=False,
                high_cardinality=True,
                event_like=False,
                unique_ratio=1.0
            ),
            missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=0),
            flags=["high_cardinality", "possible_id"]
        )
        assert analysis.cardinality == 1000000
    
    def test_high_missing_percentage(self):
        """100% missing should be valid"""
        analysis = ColumnAnalysis(
            name="sparse_col",
            type="numeric",
            missing=MissingInfo(count=100, percent=100.0),
            cardinality=0,
            unique_ratio=0.0,
            stats=ColumnStats(density=0.0),
            signals=ColumnSignals(
                constant=False,
                possible_id=False,
                high_missing=True,
                moderate_missing=False,
                high_cardinality=False,
                event_like=False,
                unique_ratio=0.0
            ),
            missing_pattern=MissingPattern(transitions=0, max_consecutive_missing=100),
            flags=["high_missing"]
        )
        assert analysis.missing.percent == 100.0
    
    def test_negative_correlation(self):
        """Negative correlations should be allowed"""
        corr = Correlation(
            feature_1="temp",
            feature_2="ice_cream_sales",
            correlation=-0.92,
            insight="strong negative correlation"
        )
        assert corr.correlation == -0.92
    
    def test_zero_variance(self):
        """Zero variance should be valid"""
        stats = ColumnStats(
            density=1.0,
            variance=0.0,
            mean=5.0,
            std=0.0
        )
        assert stats.variance == 0.0