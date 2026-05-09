import pandas as pd
import os
from pathlib import Path
import pyarrow.parquet as pq
from zipfile import BadZipFile
import logging

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SecurityError(Exception):
    """Raised when path validation fails for security reasons"""
    pass


class Ingestor:

    logger = logging.getLogger(__name__)
    
    def __init__(self, filepath: str, base_dir: str = None):

        if base_dir is None:
            base_dir = os.getcwd()
        
        self.base_dir = Path(base_dir).resolve()
        

        self.filepath = self._validate_safe_path(filepath)
        

        self._check_format()
        self._validate_exists()
    
    def _validate_safe_path(self, user_path: str) -> Path:
        """Validate user_path is within base_dir"""
        if not user_path or not user_path.strip():
            raise ValueError("Path cannot be empty or whitespace")
        
        target = (self.base_dir / user_path).resolve()
        
        if target == self.base_dir:
            raise ValueError(f"Path resolves to base directory itself: {target}")
        
        if self.base_dir not in target.parents:
            raise SecurityError(
                f"Path traversal detected: '{user_path}' resolves to '{target}', "
                f"which is outside base directory '{self.base_dir}'"
            )
        
        return target
    
    def _validate_exists(self):
        """Check if file exists"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
    
    def _check_format(self):
        """Validate file extension"""
        supported = {".csv", ".parquet", ".json", ".xls", ".xlsx"}
        if self.filepath.suffix not in supported:
            raise ValueError(
                f"Unsupported file format: {self.filepath.suffix}. "
                f"Supported: {', '.join(supported)}"
            )
    
    def load_data(self, limit=None, batch_size=None):
        """Stream data from file in batches"""
        if limit is not None and limit <= 0:
            raise ValueError("Limit must be a positive integer")
        
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be positive int")

        
        effective_batch = batch_size or 10_000
        
        # Route to appropriate streaming method
        if self.filepath.suffix == ".csv":
            gen = self._stream_csv(effective_batch)
        elif self.filepath.suffix == ".parquet":
            gen = self._stream_parquet(effective_batch)
        elif self.filepath.suffix == ".json":
            # Check file size for JSON
            file_size = self.filepath.stat().st_size
            max_size = 500 * 1024 * 1024  # 500MB
            if file_size >= max_size:
                raise ValueError(
                    f"JSON file too large: {file_size / 1024 / 1024:.1f}MB. "
                    f"Max: {max_size / 1024 / 1024:.1f}MB"
                )
            gen = self._stream_json(effective_batch)
        elif self.filepath.suffix in {".xls", ".xlsx"}:
            gen = self._stream_excel(effective_batch)
        else:
            raise ValueError(f"Unsupported format: {self.filepath.suffix}")
        
        yield from self._apply_constraints(gen, limit)
    
    def _stream_csv(self, batch_size):
        """Stream CSV file in chunks"""
        try:
            reader = pd.read_csv(
                self.filepath, 
                chunksize=batch_size,
                on_bad_lines='skip'
            )
            for chunk in reader:
                yield chunk
        except pd.errors.ParserError as e:
            self.logger.error(f"CSV parsing error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error streaming CSV: {e}")
            raise
    
    def _stream_parquet(self, batch_size):
        """Stream Parquet file in chunks"""
        try:
            parquet_file = pq.ParquetFile(self.filepath)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                yield batch.to_pandas()
        except pq.lib.ArrowInvalid as e:
            self.logger.error(f"Invalid Parquet file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error streaming Parquet: {e}")
            raise
    
    def _stream_json(self, batch_size):
        """Stream JSON file (newline-delimited or standard)"""
        # Try newline-delimited first
        try:
            reader = pd.read_json(self.filepath, lines=True, chunksize=batch_size)
            for chunk in reader:
                yield chunk
        except ValueError:
            # Not newline-delimited, try standard JSON
            self.logger.info("Not JSON-L format, loading as standard JSON")
            try:
                df = pd.read_json(self.filepath)
                for i in range(0, len(df), batch_size):
                    yield df.iloc[i:i + batch_size]
            except Exception as e:
                self.logger.error(f"Error reading JSON: {e}")
                raise
    
    def _stream_excel(self, batch_size):
        """Stream Excel file in chunks"""
        from openpyxl import load_workbook
        from itertools import islice
        
        wb = None
        try:
            wb = load_workbook(
                filename=str(self.filepath),  # Convert Path to string
                read_only=True, 
                data_only=True
            )
            
            ws = wb.active
            if ws is None:
                raise ValueError("Excel file has no active worksheet")
            
            data_iter = ws.values
            
            # Get headers
            try:
                cols = next(data_iter)
            except StopIteration:
                self.logger.warning("Excel file has no data")
                return
            
            # Stream data in batches
            while True:
                batch = list(islice(data_iter, batch_size))
                if not batch:
                    break
                yield pd.DataFrame(batch, columns=cols)
        
        except BadZipFile as e:
            self.logger.error(f"Corrupted Excel file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading Excel: {e}")
            raise
        finally:
            if wb is not None:
                wb.close()
    
    def _apply_constraints(self, generator, limit):
        """Apply row limit to generator"""
        count = 0
        for batch in generator:
            if limit and count + len(batch) > limit:
                yield batch.head(limit - count)
                return
            yield batch
            count += len(batch)