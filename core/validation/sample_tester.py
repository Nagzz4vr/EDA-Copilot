import pandas as pd
from pandas import DataFrame
from typing import List, Dict, Any

class SampleTester:

    def __init__(self, transform_executor, logger=None):
        self.transform_executor = transform_executor
        self.logger = logger
        self.violations = []

    def apply(self, df: DataFrame, transforms: list, sample_size: int = 500) -> DataFrame:
        self.violations = []
        sample_size = min(sample_size, len(df))
        sample = df.sample(n=sample_size, random_state=42).copy()

        for transform in transforms:
            try:
                sample = self.transform_executor.apply_single_transform(sample, transform)
            except Exception as e:
                violation = {
                    "transform": transform.get("id", "UNKNOWN"),
                    "operation": transform.get("operation", "UNKNOWN"),
                    "error": str(e),
                    "severity": "CRITICAL"
                }
                self.violations.append(violation)

                if self.logger:
                    self.logger.log(
                        tool="SAMPLE_TESTER",
                        intent="Sample transform failed",
                        inputs={"transform": transform},
                        outputs={"error": str(e)},
                        confidence=0.0
                    )
        return sample

    def has_data_loss(self, df_before: DataFrame, df_after: DataFrame, row_loss_threshold: float = 0.05) -> bool:
        row_loss = len(df_after) < len(df_before) * (1 - row_loss_threshold)
        col_loss = len(df_after.columns) < len(df_before.columns)
        return row_loss or col_loss

    def get_violations(self) -> List[Dict[str, Any]]:
        return self.violations