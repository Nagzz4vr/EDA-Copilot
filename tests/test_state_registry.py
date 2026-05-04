import pytest
import os
import json
import tempfile
import shutil
from core.cache.state_registry import StateRegistry


class TestStateRegistry:
    """Test suite for StateRegistry class"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def registry(self, temp_cache_dir):
        """Create a StateRegistry instance with temporary directory"""
        return StateRegistry(cache_dir=temp_cache_dir)
    
    @pytest.fixture
    def sample_entries(self, temp_cache_dir):
        """Create sample cache entries for testing"""
        entries = [
            {
                "state_uuid": "uuid-001",
                "fingerprint": "fp-alpha",
                "validation_score": 0.95,
                "dag": {"nodes": ["A", "B"]}
            },
            {
                "state_uuid": "uuid-002",
                "fingerprint": "fp-alpha",
                "validation_score": 0.85,
                "dag": {"nodes": ["C", "D"]}
            },
            {
                "state_uuid": "uuid-003",
                "fingerprint": "fp-beta",
                "validation_score": 0.75,
                "dag": {"nodes": ["E", "F"]}
            },
            {
                "state_uuid": "uuid-004",
                "fingerprint": "fp-alpha",
                "validation_score": 0.90,
                "dag": {"nodes": ["G", "H"]}
            }
        ]
        
        # Write entries to files
        for entry in entries:
            filepath = os.path.join(temp_cache_dir, f"{entry['state_uuid']}.json")
            with open(filepath, 'w') as f:
                json.dump(entry, f)
        
        return entries
    
    # Initialization Tests
    
    def test_init_creates_cache_directory(self, temp_cache_dir):
        """Test that initialization creates the cache directory"""
        new_dir = os.path.join(temp_cache_dir, "new_cache")
        assert not os.path.exists(new_dir)
        
        registry = StateRegistry(cache_dir=new_dir)
        
        assert os.path.exists(new_dir)
        assert os.path.isdir(new_dir)
        assert registry.cache_dir == new_dir
    
    def test_init_with_existing_directory(self, temp_cache_dir):
        """Test initialization with existing directory doesn't raise error"""
        # Directory already exists
        registry = StateRegistry(cache_dir=temp_cache_dir)
        
        assert registry.cache_dir == temp_cache_dir
        assert os.path.exists(temp_cache_dir)
    
    def test_init_default_cache_directory(self):
        """Test initialization with default cache directory name"""
        registry = StateRegistry()
        
        assert registry.cache_dir == "cache_storage"
        assert os.path.exists("cache_storage")
        
        # Cleanup
        shutil.rmtree("cache_storage", ignore_errors=True)
    
    # find_one Tests
    
    def test_find_one_existing_uuid(self, registry, sample_entries):
        """Test finding an existing entry by UUID"""
        result = registry.find_one({"state_uuid": "uuid-001"})
        
        assert result is not None
        assert result["state_uuid"] == "uuid-001"
        assert result["fingerprint"] == "fp-alpha"
        assert result["validation_score"] == pytest.approx(0.95)
    
    def test_find_one_non_existing_uuid(self, registry, sample_entries):
        """Test finding a non-existent UUID returns None"""
        result = registry.find_one({"state_uuid": "uuid-999"})
        
        assert result is None
    
    def test_find_one_empty_cache(self, registry):
        """Test finding in empty cache returns None"""
        result = registry.find_one({"state_uuid": "any-uuid"})
        
        assert result is None
    
    def test_find_one_returns_correct_data(self, registry, temp_cache_dir):
        """Test that find_one returns complete and correct data"""
        expected_data = {
            "state_uuid": "test-uuid",
            "fingerprint": "test-fp",
            "custom_field": "custom_value",
            "nested": {"data": [1, 2, 3]}
        }
        
        filepath = os.path.join(temp_cache_dir, "test-uuid.json")
        with open(filepath, 'w') as f:
            json.dump(expected_data, f)
        
        result = registry.find_one({"state_uuid": "test-uuid"})
        
        assert result == expected_data
    
    def test_find_one_with_empty_query(self, registry):
        """Test find_one with empty query"""
        result = registry.find_one({})
        
        assert result is None
    
    def test_find_one_with_none_uuid(self, registry):
        """Test find_one when UUID is None"""
        result = registry.find_one({"state_uuid": None})
        
        assert result is None
    
    # find_many Tests
    
    def test_find_many_matching_fingerprint(self, registry, sample_entries):
        """Test finding multiple entries with matching fingerprint"""
        results = registry.find_many({"fingerprint": "fp-alpha"})
        
        assert len(results) == 3
        fingerprints = [r["fingerprint"] for r in results]
        assert all(fp == "fp-alpha" for fp in fingerprints)
    
    def test_find_many_single_match(self, registry, sample_entries):
        """Test finding entries when only one matches"""
        results = registry.find_many({"fingerprint": "fp-beta"})
        
        assert len(results) == 1
        assert results[0]["state_uuid"] == "uuid-003"
    
    def test_find_many_no_matches(self, registry, sample_entries):
        """Test finding with no matching fingerprint"""
        results = registry.find_many({"fingerprint": "fp-nonexistent"})
        
        assert len(results) == 0
        assert results == []
    
    def test_find_many_empty_cache(self, registry):
        """Test find_many on empty cache"""
        results = registry.find_many({"fingerprint": "any-fp"})
        
        assert len(results) == 0
        assert results == []
    
    def test_find_many_without_sort(self, registry, sample_entries):
        """Test find_many without sorting returns all matches"""
        results = registry.find_many({"fingerprint": "fp-alpha"}, sort_by=None)
        
        assert len(results) == 3
        # Order is not guaranteed without sorting
        uuids = {r["state_uuid"] for r in results}
        assert uuids == {"uuid-001", "uuid-002", "uuid-004"}
    
    def test_find_many_with_sort_descending(self, registry, sample_entries):
        """Test find_many with sorting by validation_score (descending)"""
        results = registry.find_many(
            {"fingerprint": "fp-alpha"}, 
            sort_by="validation_score"
        )
        
        assert len(results) == 3
        # Should be sorted in descending order
        assert results[0]["validation_score"] ==  pytest.approx(0.95)
        assert results[1]["validation_score"] ==  pytest.approx(0.90)
        assert results[2]["validation_score"] ==  pytest.approx(0.85)
    
    def test_find_many_sort_by_different_field(self, registry, temp_cache_dir):
        """Test sorting by different fields"""
        entries = [
            {"state_uuid": "a", "fingerprint": "fp-x", "timestamp": 100},
            {"state_uuid": "b", "fingerprint": "fp-x", "timestamp": 300},
            {"state_uuid": "c", "fingerprint": "fp-x", "timestamp": 200}
        ]
        
        for entry in entries:
            filepath = os.path.join(temp_cache_dir, f"{entry['state_uuid']}.json")
            with open(filepath, 'w') as f:
                json.dump(entry, f)
        
        results = registry.find_many({"fingerprint": "fp-x"}, sort_by="timestamp")
        
        assert results[0]["timestamp"] == 300
        assert results[1]["timestamp"] == 200
        assert results[2]["timestamp"] == 100
    
    def test_find_many_sort_missing_field(self, registry, temp_cache_dir):
        """Test sorting when some entries don't have the sort field"""
        entries = [
            {"state_uuid": "a", "fingerprint": "fp-y", "score": 50},
            {"state_uuid": "b", "fingerprint": "fp-y"},  # Missing score
            {"state_uuid": "c", "fingerprint": "fp-y", "score": 30}
        ]
        
        for entry in entries:
            filepath = os.path.join(temp_cache_dir, f"{entry['state_uuid']}.json")
            with open(filepath, 'w') as f:
                json.dump(entry, f)
        
        results = registry.find_many({"fingerprint": "fp-y"}, sort_by="score")
        
        # Entry without score gets default value 0, so should be last
        assert len(results) == 3
        assert results[0]["score"] == 50
        assert results[1]["score"] == 30
        assert "score" not in results[2]
    
    def test_find_many_with_empty_query(self, registry, sample_entries):
        """Test find_many with empty query (no fingerprint)"""
        results = registry.find_many({})
        
        # Should return empty list since fingerprint is None
        assert results == []
    
    # Edge Cases and Error Handling
    
    def test_corrupted_json_file(self, registry, temp_cache_dir):
        """Test handling of corrupted JSON files"""
        # Create a corrupted JSON file
        corrupted_file = os.path.join(temp_cache_dir, "corrupted.json")
        with open(corrupted_file, 'w') as f:
            f.write("{ invalid json content")
        
        # This should raise an error when trying to read
        with pytest.raises(json.JSONDecodeError):
            registry.find_one({"state_uuid": "corrupted"})
    
    def test_empty_json_file(self, registry, temp_cache_dir):
        """Test handling of empty JSON files"""
        empty_file = os.path.join(temp_cache_dir, "empty.json")
        with open(empty_file, 'w') as f:
            f.write("")
        
        with pytest.raises(json.JSONDecodeError):
            registry.find_one({"state_uuid": "empty"})
    
    def test_find_many_with_non_json_files(self, registry, temp_cache_dir):
        """Test find_many ignores or handles non-JSON files"""
        # Create a valid entry
        valid_entry = {
            "state_uuid": "valid",
            "fingerprint": "fp-test",
            "data": "valid"
        }
        with open(os.path.join(temp_cache_dir, "valid.json"), 'w') as f:
            json.dump(valid_entry, f)
        
        # Create a text file
        with open(os.path.join(temp_cache_dir, "textfile.txt"), 'w') as f:
            f.write("not json")
        
        # Should raise error when encountering non-JSON file
        with pytest.raises(json.JSONDecodeError):
            registry.find_many({"fingerprint": "fp-test"})
    
    def test_unicode_content(self, registry, temp_cache_dir):
        """Test handling of Unicode content in JSON"""
        unicode_entry = {
            "state_uuid": "unicode-test",
            "fingerprint": "fp-unicode",
            "content": "Hello 世界 🌍 café"
        }
        
        filepath = os.path.join(temp_cache_dir, "unicode-test.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(unicode_entry, f, ensure_ascii=False)
        
        result = registry.find_one({"state_uuid": "unicode-test"})
        
        assert result is not None
        assert result["content"] == "Hello 世界 🌍 café"
    
    def test_large_json_file(self, registry, temp_cache_dir):
        """Test handling of large JSON files"""
        large_entry = {
            "state_uuid": "large-test",
            "fingerprint": "fp-large",
            "data": ["item"] * 10000  # Large array
        }
        
        filepath = os.path.join(temp_cache_dir, "large-test.json")
        with open(filepath, 'w') as f:
            json.dump(large_entry, f)
        
        result = registry.find_one({"state_uuid": "large-test"})
        
        assert result is not None
        assert len(result["data"]) == 10000
    
    # Integration Tests
    
    def test_find_one_and_find_many_consistency(self, registry, sample_entries):
        """Test that find_one and find_many return consistent data"""
        # Get entry via find_one
        entry_one = registry.find_one({"state_uuid": "uuid-001"})
        
        # Get entries via find_many
        entries_many = registry.find_many({"fingerprint": "fp-alpha"})
        entry_from_many = next(
            (e for e in entries_many if e["state_uuid"] == "uuid-001"), 
            None
        )
        
        assert entry_one == entry_from_many
    
    def test_multiple_registries_same_directory(self, temp_cache_dir, sample_entries):
        """Test multiple registry instances accessing same directory"""
        registry1 = StateRegistry(cache_dir=temp_cache_dir)
        registry2 = StateRegistry(cache_dir=temp_cache_dir)
        
        result1 = registry1.find_one({"state_uuid": "uuid-001"})
        result2 = registry2.find_one({"state_uuid": "uuid-001"})
        
        assert result1 == result2
    
    def test_query_performance_with_many_files(self, registry, temp_cache_dir):
        """Test performance with many cache files"""
        # Create 100 entries
        for i in range(100):
            entry = {
                "state_uuid": f"uuid-{i:03d}",
                "fingerprint": f"fp-{i % 10}",  # 10 different fingerprints
                "validation_score": i * 0.01
            }
            filepath = os.path.join(temp_cache_dir, f"uuid-{i:03d}.json")
            with open(filepath, 'w') as f:
                json.dump(entry, f)
        
        # Should find ~10 entries with same fingerprint
        results = registry.find_many({"fingerprint": "fp-5"}, sort_by="validation_score")
        
        assert len(results) == 10
        assert results[0]["validation_score"] == pytest.approx(0.95)
        assert results[-1]["validation_score"] ==  pytest.approx(0.05)
    
    def test_special_characters_in_uuid(self, registry, temp_cache_dir):
        """Test handling UUIDs with special characters"""
        # Note: In practice, file systems may have restrictions
        special_uuid = "uuid-with-dash_underscore"
        entry = {
            "state_uuid": special_uuid,
            "fingerprint": "fp-special",
            "data": "test"
        }
        
        filepath = os.path.join(temp_cache_dir, f"{special_uuid}.json")
        with open(filepath, 'w') as f:
            json.dump(entry, f)
        
        result = registry.find_one({"state_uuid": special_uuid})
        
        assert result is not None
        assert result["state_uuid"] == special_uuid


class TestStateRegistryEdgeCases:
    """Additional edge case tests"""
    
    def test_sort_with_none_values(self):
        """Test sorting when field values are None"""
        temp_dir = tempfile.mkdtemp()
        try:
            registry = StateRegistry(cache_dir=temp_dir)
            
            entries = [
                {"state_uuid": "a", "fingerprint": "fp", "score": None},
                {"state_uuid": "b", "fingerprint": "fp", "score": 50},
                {"state_uuid": "c", "fingerprint": "fp", "score": None}
            ]
            
            for entry in entries:
                filepath = os.path.join(temp_dir, f"{entry['state_uuid']}.json")
                with open(filepath, 'w') as f:
                    json.dump(entry, f)
            
            results = registry.find_many({"fingerprint": "fp"}, sort_by="score")
            
            # Entries with None should be treated as 0 and come last
            assert len(results) == 3
            assert results[0]["score"] == 50
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_find_many_preserves_all_fields(self):
        """Test that find_many preserves all fields from JSON"""
        temp_dir = tempfile.mkdtemp()
        try:
            registry = StateRegistry(cache_dir=temp_dir)
            
            entry = {
                "state_uuid": "test",
                "fingerprint": "fp",
                "field1": "value1",
                "field2": 123,
                "field3": {"nested": "object"},
                "field4": [1, 2, 3]
            }
            
            filepath = os.path.join(temp_dir, "test.json")
            with open(filepath, 'w') as f:
                json.dump(entry, f)
            
            results = registry.find_many({"fingerprint": "fp"})
            
            assert len(results) == 1
            assert results[0] == entry
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
