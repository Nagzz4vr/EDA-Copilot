import pandas as pd
from typing import List, Dict, Any

class DataDiff:
    def __init__(self):
        self.columns_changed=[]
        self.columns_dropped=[]
        self.columns_added=[]

        self.sample_bfr=None
        self.sample_aftr=None

        self.stats_delta={}


    def render_sampple_table(self)->str:
        return (
            f"=== BEFORE ===\n{self.sample_before}\n\n"
            f"=== AFTER ===\n{self.sample_after}"
        )
    
    def render_stats_delta(self)->str:
        lines=[]
        for col, stats in self.stats_delta.items():
            lines.append(f"\nColumn: {col}")

            for k, v in stats.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)
    

class DiffViewer:
    def compute(self,df_before: pd.DataFrame,transforms: List[Dict[str, Any]],sample_size: int = 500) -> DataDiff:
        if df_before.empty:
            raise ValueError("Input dataframe is empty")
        
        
        sample = df_before.sample(min(sample_size, len(df_before)),random_state=42).copy()
        df_after=sample.copy()

        for transform in transforms:
            try:
                df_after = self._apply_single_transform(df_after,transform)

            except Exception as e:
                print(
                    f"[DIFF_VIEWER] Transform failed during preview: {e}"
                )

                break

        diff=DataDiff()
        diff.sample_bfr=sample.head(10)
        diff.sample_aftr=sample.haed(10)
        diff.columns_changed = []
        common_cols = set(sample.columns).intersection(df_after.columns)

        for col in common_cols:
            before = sample[col].reset_index(drop=True)
            after = df_after[col].reset_index(drop=True)

            if not before.equals(after):
                diff.columns_changed.append(col)

        for c in sample.columns:
            if c not in df_after.columns:
                diff.columns_dropped.append(c)

        for c in df_after.columns:
            if c not in sample.columns:
                diff.columns_added.append(c)
        
        diff.stats_delta = self._compute_stats_delta(
            sample,
            df_after
        )

        return diff
    
    def _apply_single_transform(self,df: pd.DataFrame,transform: Dict[str, Any]) -> pd.DataFrame:
        operation = transform["operation"]
        if operation == "impute":
            strategy = transform["params"]["strategy"]
            col = transform["column"]
            if col not in df.columns:
                return df
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                mode_vals = df[col].mode()
                if not mode_vals.empty:
                    df[col] = df[col].fillna(mode_vals[0])

        elif operation == "drop_column":
            df = df.drop(
                columns=[transform["column"]],
                errors="ignore"
            )

        elif operation == "encode":
            method = transform["params"]["method"]
            col = transform["column"]
            if col not in df.columns:
                return df
            if method == "ordinal":
                unique_vals = sorted(
                    [v for v in df[col].dropna().unique()]
                )
                mapping = {
                    v: i
                    for i, v in enumerate(unique_vals)
                }
                df[col] = df[col].map(mapping)
        return df
    
    def _compute_stats_delta(self,df_before: pd.DataFrame,df_after: pd.DataFrame) -> dict:
        delta = {}
        all_columns = set(df_before.columns).union(df_after.columns)
        for col in all_columns:
            col_delta = {}
            if (col in df_before.columns and col in df_after.columns):
                before_col = df_before[col]
                after_col = df_after[col]
                col_delta["nulls_before"] = int(before_col.isna().sum())
                col_delta["nulls_after"] = int(after_col.isna().sum())
                col_delta["nulls_delta"] = (col_delta["nulls_after"]- col_delta["nulls_before"])
                col_delta["cardinality_before"] = int(before_col.nunique(dropna=True))
                col_delta["cardinality_after"] = int(after_col.nunique(dropna=True))
                col_delta["dtype_before"] = str(before_col.dtype)
                col_delta["dtype_after"] = str(after_col.dtype)
                if pd.api.types.is_numeric_dtype(after_col):
                    col_delta["mean_before"] = float(before_col.mean())
                    col_delta["mean_after"] = float(after_col.mean())
                    col_delta["std_before"] = float(before_col.std())
                    col_delta["std_after"] = float(after_col.std())
            elif col in df_before.columns:
                col_delta["status"] = "DROPPED"
            else:
                col_delta["status"] = "ADDED"
            delta[col] = col_delta
        return delta