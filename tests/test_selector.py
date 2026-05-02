import pytest
import pandas as pd
from pathlib import Path
from core.ingestion.target_variable_selector import TargetVariableSelector, SecurityError

@pytest.fixture
def sample_files_dir(tmp_path):
    """Fixture to create various file types for header extraction testing."""
    df = pd.DataFrame({
        'user_id': [1, 2], 
        'target_event': [True, False], 
        'timestamp': ['2023-01-01', '2023-01-02']
    })
    
    df.to_csv(tmp_path / "headers.csv", index=False)
    df.to_json(tmp_path / "headers.json", orient='records', lines=True)
    df.to_parquet(tmp_path / "headers.parquet")
    df.to_excel(tmp_path / "headers.xlsx", index=False)
    
    return tmp_path

class TestTargetVariableSelector:

    # --- Initialization Tests ---
    
    def test_initialization_security(self, sample_files_dir):
        # Test path traversal prevention specifically for the selector
        with pytest.raises(SecurityError, match="Path traversal detected"):
            TargetVariableSelector("../../secret.csv", base_dir=sample_files_dir)

    # --- Column Extraction Tests ---

    def test_load_column_names_csv(self, sample_files_dir):
        selector = TargetVariableSelector("headers.csv", base_dir=sample_files_dir)
        columns = selector.load_column_names()
        
        assert columns == ['user_id', 'target_event', 'timestamp']

    def test_load_column_names_json(self, sample_files_dir):
        selector = TargetVariableSelector("headers.json", base_dir=sample_files_dir)
        columns = selector.load_column_names()
        
        assert columns == ['user_id', 'target_event', 'timestamp']

    def test_load_column_names_parquet(self, sample_files_dir):
        selector = TargetVariableSelector("headers.parquet", base_dir=sample_files_dir)
        columns = selector.load_column_names()
        
        # Parquet might preserve index depending on how it's written, 
        # but for basic pandas serialization without index, it should match.
        assert set(columns).issuperset({'user_id', 'target_event', 'timestamp'})

    def test_load_column_names_excel(self, sample_files_dir):
        selector = TargetVariableSelector("headers.xlsx", base_dir=sample_files_dir)
        columns = selector.load_column_names()
        
        assert columns == ['user_id', 'target_event', 'timestamp']