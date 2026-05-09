from __future__ import annotations
 
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional
 
import pandas as pd

OutputFormat = Literal["csv", "parquet", "json"]

@dataclass
class WriterConfig:
    output_dir:    str         = "outputs/datasets"
    format:        OutputFormat = "parquet"
    csv_index:     bool        = False
    parquet_engine: str        = "pyarrow"      
    parquet_compression: str   = "snappy"
    json_orient:   str         = "records"
    overwrite:     bool        = True


@dataclass
class WriteManifest:
    job_id:       str
    file_path:    str
    format:       str
    rows_written: int
    columns:      list
    size_bytes:   int
    written_at:   str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict:
        return {
            "job_id":       self.job_id,
            "file_path":    self.file_path,
            "format":       self.format,
            "rows_written": self.rows_written,
            "columns":      self.columns,
            "size_bytes":   self.size_bytes,
            "written_at":   self.written_at,
        }
class DatasetWriter:
    def __init__(self, config: Optional[WriterConfig] = None):
        self._cfg = config or WriterConfig()

    def write(self,execution_result: Dict[str, Any],job_id: str,format_override: Optional[OutputFormat] = None,) -> WriteManifest:
        df    = self._extract_dataframe(execution_result)
        fmt   = format_override or self._cfg.format
        path  = self._build_output_path(job_id, fmt)

        self._ensure_dir(path)
        self._validate(df, job_id)
        self._write_dataframe(df, path, fmt)

        return WriteManifest(
            job_id       = job_id,
            file_path    = str(path),
            format       = fmt,
            rows_written = len(df),
            columns      = df.columns.tolist(),
            size_bytes   = os.path.getsize(path),
        )
    

    def write_all_formats(self,execution_result: Dict[str, Any],job_id: str,) -> Dict[str, WriteManifest]:
        results = {}
        for fmt in ("parquet", "csv", "json"):
            results[fmt] = self.write(execution_result, job_id, format_override=fmt)
        return results

    def _extract_dataframe(self, execution_result: Dict[str, Any]) -> pd.DataFrame:
        if "dataframe" in execution_result:
            df = execution_result["dataframe"]
            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"execution_result['dataframe'] must be pd.DataFrame, "
                    f"got {type(df).__name__}"
                )
            return df

        if "dataframe_path" in execution_result:
            path = Path(execution_result["dataframe_path"])
            if not path.exists():
                raise FileNotFoundError(
                    f"DatasetWriter: checkpoint file not found: {path}"
                )
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            if path.suffix == ".csv":
                return pd.read_csv(path)
            raise ValueError(f"Unsupported checkpoint format: {path.suffix}")

        raise KeyError(
            "execution_result must contain either 'dataframe' or 'dataframe_path'"
        )
    
    def _build_output_path(self, job_id: str, fmt: str) -> Path:
        base = Path(self._cfg.output_dir) / job_id
        return base.with_suffix(f".{fmt}")

    def _ensure_dir(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self._cfg.overwrite:
            raise FileExistsError(
                f"Output file already exists and overwrite=False: {path}"
            )

    @staticmethod
    def _validate(df: pd.DataFrame, job_id: str) -> None:
        if df.empty:
            raise ValueError(
                f"DatasetWriter: execution_result produced an empty DataFrame "
                f"for job {job_id!r}. Nothing to write."
            )

    def _write_dataframe(self, df: pd.DataFrame, path: Path, fmt: str) -> None:
        if fmt == "csv":
            df.to_csv(path, index=self._cfg.csv_index)

        elif fmt == "parquet":
            df.to_parquet(
                path,
                engine      = self._cfg.parquet_engine,
                compression = self._cfg.parquet_compression,
                index       = False,
            )

        elif fmt == "json":
            df.to_json(path, orient=self._cfg.json_orient, indent=2)

        else:
            raise ValueError(f"Unsupported output format: {fmt!r}")