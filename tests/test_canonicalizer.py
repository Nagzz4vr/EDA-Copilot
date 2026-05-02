import pytest
import numpy as np
import pandas as pd
import json
import hashlib
from copy import deepcopy
from core.ingestion.canonicalizer import Canonicalizer


class TestSanitizeJson:
    """Test suite for _sanitize_json method"""
    
    def test_sanitize_nan_to_none(self):
        """NaN values should be converted to None"""
        data = {"value": float('nan')}
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json["value"] is None
    
    def test_sanitize_inf_to_none(self):
        """Infinity values should be converted to None"""
        data = {"value": float('inf')}
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json["value"] is None
    
    def test_sanitize_negative_inf_to_none(self):
        """Negative infinity values should be converted to None"""
        data = {"value": float('-inf')}
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json["value"] is None
    
    def test_sanitize_numpy_float(self):
        """NumPy floats should be converted to Python floats with 6 significant digits"""
        data = {"value": np.float64(123.456789)}
        canonicalizer = Canonicalizer(data)
        assert isinstance(canonicalizer.raw_json["value"], float)
        assert canonicalizer.raw_json["value"] == pytest.approx(123.457, rel=1e-3)
    
    def test_sanitize_numpy_integer(self):
        """NumPy integers should be converted to Python ints"""
        data = {"value": np.int64(42)}
        canonicalizer = Canonicalizer(data)
        assert isinstance(canonicalizer.raw_json["value"], int)
        assert canonicalizer.raw_json["value"] == 42
    
    def test_sanitize_numpy_bool(self):
        """NumPy bools should be converted to Python bools"""
        data = {"value": np.bool_(True)}
        canonicalizer = Canonicalizer(data)
        assert isinstance(canonicalizer.raw_json["value"], bool)
        assert canonicalizer.raw_json["value"] is True
    
    def test_sanitize_pandas_na(self):
        """Pandas NA values should be converted to None"""
        data = {"value": pd.NA}
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json["value"] is None
    
    def test_sanitize_nested_dict(self):
        """Should recursively sanitize nested dictionaries"""
        data = {
            "outer": {
                "inner": float('nan'),
                "value": np.int64(10)
            }
        }
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json["outer"]["inner"] is None
        assert canonicalizer.raw_json["outer"]["value"] == 10
    
    def test_sanitize_nested_list(self):
        """Should recursively sanitize lists"""
        data = {"values": [float('nan'), np.int64(5), float('inf')]}
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json["values"] == [None, 5, None]
    
    def test_sanitize_mixed_nested_structure(self):
        """Should handle complex nested structures"""
        data = {
            "data": [
                {"val": float('nan')},
                {"val": np.float64(3.14159)},
                [1, 2, float('inf')]
            ]
        }
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json["data"][0]["val"] is None
        assert isinstance(canonicalizer.raw_json["data"][1]["val"], float)
        assert canonicalizer.raw_json["data"][2][2] is None
    
    def test_sanitize_preserves_regular_values(self):
        """Regular Python types should pass through unchanged"""
        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None
        }
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.raw_json == data


class TestProcessCanonicalOrder:
    """Test suite for _process_canonical_order method"""
    
    def test_sort_top_level_keys(self):
        """Top-level keys should be sorted alphabetically"""
        data = {"z": 1, "a": 2, "m": 3}
        canonicalizer = Canonicalizer(data)
        keys = list(canonicalizer.json.keys())
        assert keys == ["a", "m", "z"]
    
    def test_sort_columns_by_type_then_name(self):
        """Columns should be sorted by type, then by name"""
        data = {
            "columns": [
                {"name": "z_col", "type": "numeric"},
                {"name": "a_col", "type": "text"},
                {"name": "m_col", "type": "numeric"},
                {"name": "b_col", "type": "categorical"}
            ]
        }
        canonicalizer = Canonicalizer(data)
        sorted_cols = canonicalizer.json["columns"]
        
        # Check order: categorical (b), numeric (m, z), text (a)
        assert sorted_cols[0]["name"] == "b_col"
        assert sorted_cols[1]["name"] == "m_col"
        assert sorted_cols[2]["name"] == "z_col"
        assert sorted_cols[3]["name"] == "a_col"
    
    def test_sort_flags_in_columns(self):
        """Flags within columns should be sorted"""
        data = {
            "columns": [
                {
                    "name": "col1",
                    "type": "numeric",
                    "flags": ["skewed", "has_outliers", "constant"]
                }
            ]
        }
        canonicalizer = Canonicalizer(data)
        flags = canonicalizer.json["columns"][0]["flags"]
        assert flags == ["constant", "has_outliers", "skewed"]
    
    def test_sort_stats_dict(self):
        """Stats dictionaries should have sorted keys"""
        data = {
            "columns": [
                {
                    "name": "col1",
                    "type": "numeric",
                    "stats": {"z_stat": 1, "a_stat": 2, "m_stat": 3}
                }
            ]
        }
        canonicalizer = Canonicalizer(data)
        stats_keys = list(canonicalizer.json["columns"][0]["stats"].keys())
        assert stats_keys == ["a_stat", "m_stat", "z_stat"]
    
    def test_sort_signals_dict(self):
        """Signals dictionaries should have sorted keys"""
        data = {
            "columns": [
                {
                    "name": "col1",
                    "type": "numeric",
                    "signals": {"z_sig": True, "a_sig": False, "m_sig": True}
                }
            ]
        }
        canonicalizer = Canonicalizer(data)
        signals_keys = list(canonicalizer.json["columns"][0]["signals"].keys())
        assert signals_keys == ["a_sig", "m_sig", "z_sig"]
    
    def test_sort_missing_dict(self):
        """Missing dictionaries should have sorted keys"""
        data = {
            "columns": [
                {
                    "name": "col1",
                    "type": "numeric",
                    "missing": {"percent": 10, "count": 5}
                }
            ]
        }
        canonicalizer = Canonicalizer(data)
        missing_keys = list(canonicalizer.json["columns"][0]["missing"].keys())
        assert missing_keys == ["count", "percent"]
    
    def test_empty_columns_list(self):
        """Should handle empty columns list"""
        data = {"columns": []}
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.json["columns"] == []
    
    def test_missing_columns_key(self):
        """Should handle data without columns key"""
        data = {"other_key": "value"}
        canonicalizer = Canonicalizer(data)
        assert "columns" not in canonicalizer.json
    
    def test_preserve_non_dict_non_list_values(self):
        """Non-dict, non-list values should be preserved"""
        data = {
            "columns": [
                {
                    "name": "col1",
                    "type": "numeric",
                    "stats": {"value": 42}
                }
            ]
        }
        canonicalizer = Canonicalizer(data)
        assert canonicalizer.json["columns"][0]["stats"]["value"] == 42


class TestAddStateInfo:
    """Test suite for add_state_info method"""
    
    def test_returns_correct_structure(self):
        """Should return metadata and canonical_data"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        assert "metadata" in result
        assert "canonical_data" in result
    
    def test_metadata_has_required_fields(self):
        """Metadata should contain state_uuid, fingerprint, and schema_version"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        assert "state_uuid" in result["metadata"]
        assert "fingerprint" in result["metadata"]
        assert "schema_version" in result["metadata"]
    
    def test_state_uuid_is_consistent(self):
        """Same data should produce the same state_uuid"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer1 = Canonicalizer(deepcopy(data))
        canonicalizer2 = Canonicalizer(deepcopy(data))
        
        result1 = canonicalizer1.add_state_info()
        result2 = canonicalizer2.add_state_info()
        
        assert result1["metadata"]["state_uuid"] == result2["metadata"]["state_uuid"]
    
    def test_state_uuid_changes_with_data(self):
        """Different data should produce different state_uuid"""
        data1 = {"columns": [{"name": "col1", "type": "numeric"}]}
        data2 = {"columns": [{"name": "col2", "type": "text"}]}
        
        canonicalizer1 = Canonicalizer(data1)
        canonicalizer2 = Canonicalizer(data2)
        
        result1 = canonicalizer1.add_state_info()
        result2 = canonicalizer2.add_state_info()
        
        assert result1["metadata"]["state_uuid"] != result2["metadata"]["state_uuid"]
    
    def test_fingerprint_is_consistent(self):
        """Same column structure should produce the same fingerprint"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer1 = Canonicalizer(deepcopy(data))
        canonicalizer2 = Canonicalizer(deepcopy(data))
        
        result1 = canonicalizer1.add_state_info()
        result2 = canonicalizer2.add_state_info()
        
        assert result1["metadata"]["fingerprint"] == result2["metadata"]["fingerprint"]
    
    def test_fingerprint_ignores_stats(self):
        """Fingerprint should only depend on column names and types, not stats"""
        data1 = {"columns": [{"name": "col1", "type": "numeric", "stats": {"mean": 5}}]}
        data2 = {"columns": [{"name": "col1", "type": "numeric", "stats": {"mean": 10}}]}
        
        canonicalizer1 = Canonicalizer(data1)
        canonicalizer2 = Canonicalizer(data2)
        
        result1 = canonicalizer1.add_state_info()
        result2 = canonicalizer2.add_state_info()
        
        assert result1["metadata"]["fingerprint"] == result2["metadata"]["fingerprint"]
    
    def test_fingerprint_changes_with_column_name(self):
        """Different column names should produce different fingerprints"""
        data1 = {"columns": [{"name": "col1", "type": "numeric"}]}
        data2 = {"columns": [{"name": "col2", "type": "numeric"}]}
        
        canonicalizer1 = Canonicalizer(data1)
        canonicalizer2 = Canonicalizer(data2)
        
        result1 = canonicalizer1.add_state_info()
        result2 = canonicalizer2.add_state_info()
        
        assert result1["metadata"]["fingerprint"] != result2["metadata"]["fingerprint"]
    
    def test_fingerprint_changes_with_column_type(self):
        """Different column types should produce different fingerprints"""
        data1 = {"columns": [{"name": "col1", "type": "numeric"}]}
        data2 = {"columns": [{"name": "col1", "type": "text"}]}
        
        canonicalizer1 = Canonicalizer(data1)
        canonicalizer2 = Canonicalizer(data2)
        
        result1 = canonicalizer1.add_state_info()
        result2 = canonicalizer2.add_state_info()
        
        assert result1["metadata"]["fingerprint"] != result2["metadata"]["fingerprint"]
    
    def test_custom_schema_version(self):
        """Should accept custom schema version"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info(schema_version="2.0.0")
        
        assert result["metadata"]["schema_version"] == "2.0.0"
    
    def test_default_schema_version(self):
        """Should use default schema version 1.0.0"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        assert result["metadata"]["schema_version"] == "1.0.0"
    
    def test_state_uuid_length(self):
        """state_uuid should be 32 characters (truncated SHA256)"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        assert len(result["metadata"]["state_uuid"]) == 32
    
    def test_fingerprint_length(self):
        """fingerprint should be 64 characters (full SHA256)"""
        data = {"columns": [{"name": "col1", "type": "numeric"}]}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        assert len(result["metadata"]["fingerprint"]) == 64
    
    def test_canonical_data_is_json(self):
        """canonical_data should be the processed JSON"""
        data = {"z": 1, "a": 2, "columns": []}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        # Should be sorted
        assert list(result["canonical_data"].keys()) == ["a", "columns", "z"]
    
    def test_empty_columns_fingerprint(self):
        """Should handle empty columns list for fingerprint"""
        data = {"columns": []}
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        assert "fingerprint" in result["metadata"]
        assert len(result["metadata"]["fingerprint"]) == 64


class TestCanonicalizerIntegration:
    """Integration tests for Canonicalizer"""
    
    def test_full_pipeline(self):
        """Test complete canonicalization pipeline"""
        data = {
            "z_key": "value",
            "columns": [
                {
                    "name": "numeric_col",
                    "type": "numeric",
                    "flags": ["skewed", "constant"],
                    "stats": {"z_stat": 1, "a_stat": 2},
                    "value": float('nan')
                },
                {
                    "name": "cat_col",
                    "type": "categorical",
                    "flags": ["imbalanced"],
                    "signals": {"z_sig": True, "a_sig": False}
                }
            ],
            "a_key": np.int64(42)
        }
        
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        # Check structure
        assert "metadata" in result
        assert "canonical_data" in result
        
        # Check top-level keys are sorted
        assert list(result["canonical_data"].keys()) == ["a_key", "columns", "z_key"]
        
        # Check columns are sorted
        assert result["canonical_data"]["columns"][0]["name"] == "cat_col"
        assert result["canonical_data"]["columns"][1]["name"] == "numeric_col"
        
        # Check flags are sorted
        assert result["canonical_data"]["columns"][1]["flags"] == ["constant", "skewed"]
        
        # Check NaN was sanitized
        assert result["canonical_data"]["columns"][1]["value"] is None
        
        # Check NumPy type was converted
        assert result["canonical_data"]["a_key"] == 42
        assert isinstance(result["canonical_data"]["a_key"], int)
    
    def test_deterministic_output(self):
        """Multiple runs should produce identical output"""
        data = {
            "columns": [
                {"name": "col1", "type": "numeric", "stats": {"mean": 5.5}},
                {"name": "col2", "type": "text", "flags": ["long_text"]}
            ]
        }
        
        results = []
        for _ in range(3):
            canonicalizer = Canonicalizer(deepcopy(data))
            result = canonicalizer.add_state_info()
            results.append(json.dumps(result, sort_keys=True))
        
        # All results should be identical
        assert results[0] == results[1] == results[2]
    
    def test_handles_complex_real_world_structure(self):
        """Test with realistic data structure"""
        data = {
            "dataset_overview": {
                "num_rows": 1000,
                "num_columns": 10
            },
            "dataset_health": {
                "duplicate_rows": 5,
                "memory_usage_mb": 2.5
            },
            "columns": [
                {
                    "name": "age",
                    "type": "numeric",
                    "missing": {"count": 10, "percent": 1.0},
                    "cardinality": 50,
                    "stats": {
                        "mean": 35.5,
                        "std": 12.3,
                        "min": 18.0,
                        "max": 80.0
                    },
                    "signals": {
                        "skewed": False,
                        "has_outliers": True,
                        "constant": False
                    },
                    "flags": ["has_outliers"]
                }
            ],
            "top_correlations": []
        }
        
        canonicalizer = Canonicalizer(data)
        result = canonicalizer.add_state_info()
        
        # Should complete without errors
        assert result["metadata"]["state_uuid"] is not None
        assert result["metadata"]["fingerprint"] is not None
        assert result["canonical_data"] is not None