import pytest
import pandas as pd
from pathlib import Path
from core.ingestion.ingestor import Ingestor, SecurityError

@pytest.fixture
def sample_data_dir(tmp_path):
    """Fixture to create a temporary directory with sample files."""
    df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
    
    # Create CSV
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    
    # Create JSON (Newline delimited)
    json_path = tmp_path / "data.json"
    df.to_json(json_path, orient='records', lines=True)
    
    # Create Parquet
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)
    
    # Create invalid file type
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("dummy text")
    
    return tmp_path

class TestIngestor:
    
    # --- Path and Validation Tests ---
    
    def test_empty_path(self, sample_data_dir):
        with pytest.raises(ValueError, match="Path cannot be empty or whitespace"):
            Ingestor("", base_dir=sample_data_dir)

    def test_path_resolves_to_base_dir(self, sample_data_dir):
        with pytest.raises(ValueError, match="resolves to base directory itself"):
            Ingestor(".", base_dir=sample_data_dir)

    def test_path_traversal_security(self, sample_data_dir):
        with pytest.raises(SecurityError, match="Path traversal detected"):
            # Attempt to escape the base directory
            Ingestor("../outside.csv", base_dir=sample_data_dir)

    def test_file_not_found(self, sample_data_dir):
        with pytest.raises(FileNotFoundError):
            Ingestor("nonexistent.csv", base_dir=sample_data_dir)

    def test_unsupported_format(self, sample_data_dir):
        with pytest.raises(ValueError, match="Unsupported file format"):
            Ingestor("data.txt", base_dir=sample_data_dir)

    # --- Data Loading and Constraint Tests ---

    def test_load_data_invalid_constraints(self, sample_data_dir):
        ingestor = Ingestor("data.csv", base_dir=sample_data_dir)
        
        with pytest.raises(ValueError, match="Limit must be a positive integer"):
            list(ingestor.load_data(limit=-5))
            
        with pytest.raises(ValueError, match="batch_size must be positive int"):
            list(ingestor.load_data(batch_size=0))

    def test_stream_csv_batches(self, sample_data_dir):
        ingestor = Ingestor("data.csv", base_dir=sample_data_dir)
        # Test batching: 10 rows total, batch size 3 means 4 batches (3, 3, 3, 1)
        batches = list(ingestor.load_data(batch_size=3))
        
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1
        assert list(batches[0].columns) == ['A', 'B']

    def test_stream_json(self, sample_data_dir):
        ingestor = Ingestor("data.json", base_dir=sample_data_dir)
        batches = list(ingestor.load_data(batch_size=5))
        
        assert len(batches) == 2
        assert len(batches[0]) == 5

    def test_stream_parquet(self, sample_data_dir):
        ingestor = Ingestor("data.parquet", base_dir=sample_data_dir)
        batches = list(ingestor.load_data(batch_size=4))
        
        assert len(batches) == 3

    def test_apply_limit(self, sample_data_dir):
        ingestor = Ingestor("data.csv", base_dir=sample_data_dir)
        # 10 rows total, limit to 4
        batches = list(ingestor.load_data(limit=4, batch_size=3))
        
        # Should return 2 batches: one of size 3, one of size 1
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 1
        
        # Combine batches to check total rows
        total_rows = sum(len(batch) for batch in batches)
        assert total_rows == 4