
# Auto-generated pipeline: hitl_test_001
# Generated at: 2026-05-10T13:59:23.923029+00:00
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.impute._base import SimpleImputer
from sklearn.preprocessing._encoders import OneHotEncoder,OrdinalEncoder
pipeline = Pipeline(steps=[
("impute_mean_Cabin", SimpleImputer(add_indicator=False, copy=True, keep_empty_features=False, missing_values=np.nan, strategy='mean')),
    ("label_encode_Sex", OrdinalEncoder(categories='auto', dtype=np.float64, encoded_missing_value=np.nan, handle_unknown='error')),
    ("label_encode_Survived", OrdinalEncoder(categories='auto', dtype=np.float64, encoded_missing_value=np.nan, handle_unknown='error')),
    ("one_hot_encode_Embarked", OneHotEncoder(categories='auto', dtype=np.float64, feature_name_combiner='concat', handle_unknown='ignore', sparse_output=False)),
    ("one_hot_encode_Parch", OneHotEncoder(categories='auto', dtype=np.float64, feature_name_combiner='concat', handle_unknown='ignore', sparse_output=False)),
    ("one_hot_encode_Pclass", OneHotEncoder(categories='auto', dtype=np.float64, feature_name_combiner='concat', handle_unknown='ignore', sparse_output=False)),
    ("one_hot_encode_SibSp", OneHotEncoder(categories='auto', dtype=np.float64, feature_name_combiner='concat', handle_unknown='ignore', sparse_output=False))
])