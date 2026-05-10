from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Literal

class MissingInfo(BaseModel):
    """Model for missing value statistics."""
    count: int
    percent: float

class MissingPattern(BaseModel):
    """Model for the pattern of missing values."""
    transitions: int
    max_consecutive_missing: int

class ColumnStats(BaseModel):
    """
    Model for all possible statistical measures of a column.
    Fields are optional as they depend on the column's data type.
    """
    density: float
    entropy: Optional[float] = None
    top_values: Optional[List[Any]] = None
    avg_length: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skew: Optional[float] = None
    outlier_count: Optional[int] = None
    variance: Optional[float] = None

class ColumnSignals(BaseModel):
    """Model for all boolean flags indicating data quality signals."""
    imbalanced: Optional[bool] = None
    long_text: Optional[bool] = None
    skewed: Optional[bool] = None
    has_outliers: Optional[bool] = None
    low_variance: Optional[bool] = None
    constant: bool
    possible_id: bool
    high_missing: bool
    moderate_missing: bool
    high_cardinality: bool
    event_like: bool
    unique_ratio: float
    is_target: bool = False 



class ColumnAnalysis(BaseModel):
    """The complete analytical model for a single DataFrame column."""
    name: str
    type: str
    missing: MissingInfo
    cardinality: int
    unique_ratio: float
    stats: ColumnStats
    signals: ColumnSignals
    missing_pattern: MissingPattern
    flags: List[str]


    @field_validator('flags')
    @classmethod
    def check_flags_sorted(cls, v: List[str]) -> List[str]:
        if v != sorted(v):
            raise ValueError("Flags must be sorted alphabetically")
        return v



class DatasetOverview(BaseModel):
    """Model for the basic overview of the dataset."""
    num_rows: int
    num_columns: int

class DatasetHealth(BaseModel):
    """Model for dataset-level health metrics."""
    duplicate_rows: int
    duplicate_percent: float
    memory_usage_mb: float

class Correlation(BaseModel):
    """Model for a high-correlation pair."""
    feature_1: str
    feature_2: str
    correlation: float
    insight: str



class CanonicalData(BaseModel):
    """The main model for the canonicalized data structure."""
    columns: List[ColumnAnalysis]
    dataset_health: DatasetHealth
    dataset_overview: DatasetOverview
    top_correlations: List[Correlation]


    @field_validator('columns')
    @classmethod
    def check_columns_sorted(cls, v: List[ColumnAnalysis]) -> List[ColumnAnalysis]:
        # Sorts by 'type' then by 'name' to check against the input list
        sorted_v = sorted(v, key=lambda x: (x.type, x.name))
        if any(v[i].name != sorted_v[i].name for i in range(len(v))):
            raise ValueError("Columns must be sorted by type, then by name")
        return v


class Metadata(BaseModel):
    """Model for the state and versioning metadata."""
    state_uuid: str
    fingerprint: str
    schema_version: str

class CanonicalizedOutput(BaseModel):
    """The final, complete model for the entire output including metadata."""
    metadata: Metadata
    canonical_data: CanonicalData