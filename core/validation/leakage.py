import pandas as pd
import numpy as np
import re
from scipy.stats import spearmanr, pointbiserialr
from sklearn.preprocessing import LabelEncoder

class LeakageDetector:
    def __init__(self, threshold_critical=0.95, threshold_warning=0.85):
        self.critical = threshold_critical
        self.warning = threshold_warning
        self.blacklist = [r'target', r'label', r'y_', r'_id$', r'^id_']

    def calculate_appropriate_metric(self, col_data, label_data):
        valid_idx = col_data.notna() & label_data.notna()
        x = col_data[valid_idx]
        y = label_data[valid_idx]
        x_is_num = pd.api.types.is_numeric_dtype(x)
        y_is_num = pd.api.types.is_numeric_dtype(y)

        try:
            if x_is_num and y_is_num:
                score, _ = spearmanr(x, y)
                return abs(score)
        
            elif x_is_num != y_is_num:
                cat_col = y if x_is_num else x
                num_col = x if x_is_num else y
        
                encoded_cat = LabelEncoder().fit_transform(cat_col.astype(str))
                score, _ = pointbiserialr(encoded_cat, num_col)
                return abs(score)
            else:
                return 0.0 
        except:
            return 0.0

    def matches_blacklist(self, col_name):
        return any(re.search(pattern, col_name.lower()) for pattern in self.blacklist)