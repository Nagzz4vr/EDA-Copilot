import pandas as pd
import numpy as np

class ContextBuilder:
    def __init__(self,df:pd.DataFrame,target_col: str = None):
        self.df=df
        self.target_col = target_col
        self.validate_df()

    def validate_df(self):
        """Ensures we aren't processing junk data."""
        if self.df.empty:
            raise ValueError("Empty Dataframe")
        elif self.df.dropna(how='all').empty:
            raise ValueError("Filled with NA in Dataframe")
        
        if self.target_col and self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")
        
    def build_context(self):
        context={}
        context["dataset_overview"]={"num_rows":self.df.shape[0],"num_columns":self.df.shape[1],"target_variable": self.target_col}
        context["dataset_health"]=self._dataset_level_analysis()
        context["columns"] = [{"name": col,**self._analyze_column(col)}for col in self.df.columns]
        context["top_correlations"] = self._get_high_correlations()
        return context
    
    def _dataset_level_analysis(self):
        memory_bytes = self.df.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 * 1024)
        return {
            "duplicate_rows": int(self.df.duplicated().sum()),
            "duplicate_percent": round(self.df.duplicated().mean() * 100, 2),
            "memory_usage_mb": float(memory_mb)
        }
    
    def _analyze_column(self, col_name: str) -> dict:
        full_series = self.df[col_name]
        sample = self._get_sample(full_series)
        col_type = self._classify_type(full_series)

        nulls = full_series.isnull()
        missing_ratio = nulls.mean()
        density = 1 - missing_ratio

        missing = {
            "count":int(nulls.sum()),
            "percent": round(missing_ratio * 100, 3)
        }
        cardinality = full_series.nunique()
        unique_ratio = cardinality / len(full_series) if len(full_series) > 0 else 0

        meta = {
            "type": col_type,
            "missing": missing,
            "cardinality": cardinality,
            "unique_ratio": round(unique_ratio, 3),
            "stats": {},
            "signals": {}
        }

        # --- Categorical & Text Logic ---
        if col_type == "categorical":
            meta["stats"]["entropy"] = round(self._calculate_entropy(sample), 3)
            value_counts = sample.value_counts(normalize=True, dropna=True)
            if not value_counts.empty:
                is_imbalanced = bool(value_counts.iloc[0] > 0.5)
                meta["stats"]["top_values"] = value_counts.head(3).index.tolist()
                meta["signals"]["imbalanced"] = is_imbalanced

        elif col_type == "text":
            avg_len = float(sample.dropna().astype(str).str.len().mean())
            meta["stats"]["avg_length"] = round(avg_len, 2)
            meta["signals"]["long_text"] = avg_len > 30

        # --- Numeric Logic (Move Variance inside here!) ---
        if col_type == "numeric":
            q1, q3 = sample.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = full_series[(full_series < (q1 - 1.5 * iqr)) | (full_series > (q3 + 1.5 * iqr))].count()
            skew_val = float(full_series.skew())
            variance = float(full_series.var()) 

            meta["stats"].update({
                "mean": round(float(full_series.mean()), 2),
                "std": round(float(full_series.std()), 2),
                "min": float(full_series.min()),
                "max": float(full_series.max()),
                "skew":round(skew_val, 2),
                "outlier_count": int(outliers),
                "variance": round(variance, 4)
            })
            meta["signals"]["skewed"] = abs(skew_val) > 1.5
            meta["signals"]["has_outliers"] = bool(outliers > 0)
            meta["signals"]["low_variance"] = variance < 1e-5
            

        # --- Shared Global Signals ---
   
        meta["stats"]["density"] = round(density, 3)
        meta["signals"]["is_target"] = (col_name == self.target_col)
        meta["signals"]["constant"] = cardinality == 1
        meta["signals"]["possible_id"] = (col_type in ["text", "categorical"] and unique_ratio > 0.9)
        meta["signals"]["high_missing"] = missing["percent"] > 70
        meta["signals"]["moderate_missing"]  = 20 < missing["percent"] <= 70
        meta["signals"]["high_cardinality"] = unique_ratio > 0.9
        meta["signals"]["unique_ratio"] = unique_ratio
        
        
        is_numeric_or_bool = col_type in ["numeric", "categorical"]
        meta["signals"]["event_like"] = (missing["percent"] > 50) and is_numeric_or_bool and (cardinality <= 5)

        meta["missing_pattern"] = self._missing_pattern(sample)
        
        # Populate flags list at the very end
        meta["flags"] = sorted([k for k, v in meta["signals"].items() if v])
        return meta
    
    

    def _get_high_correlations(self):
        """Identifies pairs with strong linear relationships."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2: return {}

        if numeric_df.shape[1] > 100:
            numeric_df = numeric_df.iloc[:, :100]
        if len(numeric_df) > 5000:
            numeric_df = numeric_df.sample(5000, random_state=42)

        corr_matrix = numeric_df.corr().abs()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.75:
                    high_corr.append({
                    "feature_1": corr_matrix.columns[i],
                    "feature_2": corr_matrix.columns[j],
                    "correlation": round(corr_matrix.iloc[i, j], 2),
                    "insight": "potential multicollinearity"
                    })
        high_corr.sort(key=lambda x: x["correlation"], reverse=True)
        return high_corr[:5]
    
    def _classify_type(self, series: pd.Series) -> str:

        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_extension_array_dtype(series):
            return "categorical"


        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        unique_count = series.nunique()
        total_count  = len(series)

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            ratio = unique_count / total_count
            

            if pd.api.types.is_float_dtype(series):
                return "numeric"
                
            if unique_count <= 5 or ratio < 0.10:
                return "categorical"
                
            return "numeric" 

   
        if series.dtype == "object":
            cleaned = series.dropna().astype(str)
            if len(cleaned) == 0:
                return "categorical"

            avg_len      = cleaned.str.len().mean()
            ratio_unique = cleaned.nunique() / len(cleaned)

            if avg_len > 25 and ratio_unique > 0.25:
                return "text"

        return "categorical"
    

    def _calculate_entropy(self, series):
        probs = series.value_counts(normalize=True)
        return float(-(probs * np.log2(probs + 1e-9)).sum())
    
    def _missing_pattern(self, series: pd.Series) -> dict:
        is_null = series.isnull().astype(int)
        transitions = int((is_null.diff().abs().sum()) if len(is_null) > 1 else 0)
        max_streak = 0
        current_streak = 0
        for val in is_null:
            if val == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return {
            "transitions": transitions,
            "max_consecutive_missing": max_streak
        }
    def _get_sample(self, series: pd.Series, max_size=10000):
        if len(series) > max_size:
            return series.sample(max_size, random_state=42)
        return series