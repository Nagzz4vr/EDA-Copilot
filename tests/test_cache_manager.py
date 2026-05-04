import pytest
from unittest.mock import Mock, MagicMock
from core.cache.cache_manager import CacheManager


class TestCacheManager:
    """Test suite for CacheManager class"""
    
    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage provider"""
        return Mock()
    
    @pytest.fixture
    def cache_manager(self, mock_storage):
        """Create a CacheManager instance with mock storage"""
        return CacheManager(mock_storage, "v1.0")
    
    @pytest.fixture
    def sample_package(self):
        """Sample package data for testing"""
        return {
            "metadata": {
                "state_uuid": "abc-123-def-456",
                "fingerprint": "fp-789"
            }
        }
    
    # EXACT_HIT Tests
    
    def test_exact_hit_valid_entry(self, cache_manager, mock_storage, sample_package):
        """Test exact cache hit with valid entry"""
        expected_dag = {"nodes": ["A", "B"], "edges": []}
        mock_storage.find_one.return_value = {
            "state_uuid": "abc-123-def-456",
            "dag": expected_dag,
            "rule_version": "v1.0"
        }
        
        # Mock _is_still_valid to return True
        cache_manager._is_still_valid = Mock(return_value=True)
        
        result_type, result_dag = cache_manager.lookup(sample_package)
        
        assert result_type == "EXACT_HIT"
        assert result_dag == expected_dag
        mock_storage.find_one.assert_called_once_with(
            {"state_uuid": "abc-123-def-456"}
        )
    
    def test_exact_hit_invalid_entry_falls_through_to_partial(
        self, cache_manager, mock_storage, sample_package
    ):
        """Test exact hit found but invalid, falls through to partial hit"""
        exact_entry = {
            "state_uuid": "abc-123-def-456",
            "dag": {"nodes": ["old"]},
            "rule_version": "v0.9"  # Old version
        }
        partial_dag = {"nodes": ["C", "D"], "edges": []}
        
        mock_storage.find_one.return_value = exact_entry
        mock_storage.find_many.return_value = [
            {"dag": partial_dag, "validation_score": 0.95}
        ]
        
        cache_manager._is_still_valid = Mock(return_value=False)
        
        result_type, result_dag = cache_manager.lookup(sample_package)
        
        assert result_type == "PARTIAL_HIT"
        assert result_dag == partial_dag
        cache_manager._is_still_valid.assert_called_once_with(exact_entry)
    
    def test_exact_hit_invalid_entry_no_partial(
        self, cache_manager, mock_storage, sample_package
    ):
        """Test exact hit invalid and no partial candidates available"""
        exact_entry = {"state_uuid": "abc-123-def-456", "dag": {"old": True}}
        
        mock_storage.find_one.return_value = exact_entry
        mock_storage.find_many.return_value = []
        
        cache_manager._is_still_valid = Mock(return_value=False)
        
        result_type, result_dag = cache_manager.lookup(sample_package)
        
        assert result_type == "MISS"
        assert result_dag is None
    
    # PARTIAL_HIT Tests
    
    def test_partial_hit_no_exact_match(
        self, cache_manager, mock_storage, sample_package
    ):
        """Test partial hit when no exact match exists"""
        partial_dag = {"nodes": ["E", "F"], "edges": [(0, 1)]}
        
        mock_storage.find_one.return_value = None
        mock_storage.find_many.return_value = [
            {"dag": partial_dag, "validation_score": 0.85},
            {"dag": {"nodes": ["G"]}, "validation_score": 0.70}
        ]
        
        result_type, result_dag = cache_manager.lookup(sample_package)
        
        assert result_type == "PARTIAL_HIT"
        assert result_dag == partial_dag  # Should return highest scored candidate
        mock_storage.find_many.assert_called_once_with(
            {"fingerprint": "fp-789"},
            sort_by="validation_score"
        )
    
    def test_partial_hit_returns_first_candidate(
        self, cache_manager, mock_storage, sample_package
    ):
        """Test that partial hit returns first candidate (highest score)"""
        candidates = [
            {"dag": {"best": True}, "validation_score": 0.99},
            {"dag": {"second": True}, "validation_score": 0.88},
            {"dag": {"third": True}, "validation_score": 0.77}
        ]
        
        mock_storage.find_one.return_value = None
        mock_storage.find_many.return_value = candidates
        
        result_type, result_dag = cache_manager.lookup(sample_package)
        
        assert result_type == "PARTIAL_HIT"
        assert result_dag == {"best": True}
    
    # MISS Tests
    
    def test_miss_no_exact_no_partial(
        self, cache_manager, mock_storage, sample_package
    ):
        """Test cache miss when no exact or partial matches exist"""
        mock_storage.find_one.return_value = None
        mock_storage.find_many.return_value = []
        
        result_type, result_dag = cache_manager.lookup(sample_package)
        
        assert result_type == "MISS"
        assert result_dag is None
    
    def test_miss_empty_candidates_list(
        self, cache_manager, mock_storage, sample_package
    ):
        """Test cache miss with empty candidates list"""
        mock_storage.find_one.return_value = None
        mock_storage.find_many.return_value = []
        
        result_type, result_dag = cache_manager.lookup(sample_package)
        
        assert result_type == "MISS"
        assert result_dag is None
    
    # Edge Cases
    
    def test_missing_metadata_fields(self, cache_manager, mock_storage):
        """Test behavior with missing metadata fields"""
        incomplete_package = {
            "metadata": {
                "state_uuid": "only-uuid"
                # Missing fingerprint
            }
        }
        
        mock_storage.find_one.return_value = None
        mock_storage.find_many.return_value = []
        
        # This should raise KeyError or handle gracefully depending on requirements
        with pytest.raises(KeyError):
            cache_manager.lookup(incomplete_package)
    
    def test_none_package(self, cache_manager, mock_storage):
        """Test behavior with None package"""
        with pytest.raises((TypeError, AttributeError)):
            cache_manager.lookup(None)
    
    def test_empty_package(self, cache_manager, mock_storage):
        """Test behavior with empty package"""
        with pytest.raises(KeyError):
            cache_manager.lookup({})
    
    def test_rule_version_stored_correctly(self, mock_storage):
        """Test that rule version is stored during initialization"""
        manager = CacheManager(mock_storage, "v2.5.1")
        assert manager.rule_version == "v2.5.1"
    
    def test_storage_provider_stored_correctly(self, mock_storage):
        """Test that storage provider is stored during initialization"""
        manager = CacheManager(mock_storage, "v1.0")
        assert manager.storage == mock_storage
    
    # Integration-style Tests
    
    def test_lookup_workflow_all_paths(self, cache_manager, mock_storage):
        """Test complete lookup workflow through different code paths"""
        package = {
            "metadata": {
                "state_uuid": "test-uuid",
                "fingerprint": "test-fp"
            }
        }
        
        # First call: exact hit
        mock_storage.find_one.return_value = {
            "dag": {"exact": True},
            "rule_version": "v1.0"
        }
        cache_manager._is_still_valid = Mock(return_value=True)
        
        result_type, _ = cache_manager.lookup(package)
        assert result_type == "EXACT_HIT"
        
        # Second call: partial hit (exact invalid)
        cache_manager._is_still_valid = Mock(return_value=False)
        mock_storage.find_many.return_value = [{"dag": {"partial": True}}]
        
        result_type, _ = cache_manager.lookup(package)
        assert result_type == "PARTIAL_HIT"
        
        # Third call: miss
        mock_storage.find_one.return_value = None
        mock_storage.find_many.return_value = []
        
        result_type, result_dag = cache_manager.lookup(package)
        assert result_type == "MISS"
        assert result_dag is None



