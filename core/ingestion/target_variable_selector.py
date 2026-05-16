import pandas as pd
import os
from pathlib import Path
import pyarrow.parquet as pq
import csv
import json

class SecurityError(Exception):
    """Raised when path validation fails for security reasons"""
    pass

class TargetVariableSelector:
    def __init__(self, filepath: str, base_dir: str = None):
        self.base_dir = Path(base_dir or os.getcwd()).resolve()
        self.filepath = self._validate_safe_path(filepath)
        self._check_format()
        self._validate_exists()
    
    def _validate_safe_path(self, user_path: str) -> Path:
        """Validate user_path is within base_dir"""
        if not user_path or not user_path.strip():
            raise ValueError("Path cannot be empty or whitespace")
        
        user_path_obj = Path(user_path)

        if user_path_obj.is_absolute():
            target = user_path_obj.resolve()
        else:
            target = (self.base_dir / user_path_obj).resolve()
    
        if target == self.base_dir:
            raise ValueError(
                f"Path resolves to base directory itself: {target}"
            )
    
        try:
            target.relative_to(self.base_dir)
    
        except ValueError:
            raise SecurityError(
                f"Path traversal detected: '{user_path}' resolves to '{target}', "
                f"which is outside base directory '{self.base_dir}'"
            )
    
        return target
    
    def _validate_exists(self):
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
    
    def _check_format(self):
        supported = {".csv", ".parquet", ".json", ".xls", ".xlsx"}
        if self.filepath.suffix not in supported:
            raise ValueError(f"Unsupported format: {self.filepath.suffix}")

    def load_column_names(self):
        """Extract columns without loading the entire file."""
        ext = self.filepath.suffix
        
        if ext == ".csv":
            with open(self.filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                return next(reader)
        
        elif ext == ".parquet":
            metadata = pq.read_metadata(self.filepath)
            return metadata.schema.names
            
        elif ext == ".json":
            # Note: This assumes JSON Lines (JSONL). 
            # For standard JSON arrays, consider using the 'ijson' library.
            with open(self.filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                return list(json.loads(first_line).keys())
        
        elif ext in {".xls", ".xlsx"}:
            # nrows=0 tells pandas to only read the header
            df = pd.read_excel(self.filepath, nrows=0)
            return df.columns.tolist()
            
        return []