"""Các transformer sklearn, chỉ học tham số khi pipeline.fit được gọi."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_original_preprocessor():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("scaler", MinMaxScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC_COLUMNS)),
            ("cat", categorical, list(CATEGORICAL_COLUMNS)),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def build_fe_preprocessor():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC_COLUMNS) + ["chol_per_age", "bps_per_age", "hr_ratio"]),
            ("cat", categorical, list(CATEGORICAL_COLUMNS) + ["age_bin"]),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
