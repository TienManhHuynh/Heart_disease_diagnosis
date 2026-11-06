"""Tạo đặc trưng theo công thức tài liệu; quantile tuổi học từ training subset."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class AddClinicalFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, n_age_bins=5):
        self.n_age_bins = n_age_bins

    def fit(self, X, y=None):
        if self.n_age_bins < 1:
            raise ValueError("n_age_bins phải >= 1.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        ages = pd.to_numeric(X["age"], errors="coerce").dropna()
        if ages.nunique() < 2:
            self.age_bin_edges_ = np.array([-np.inf, np.inf])
        else:
            _, edges = pd.qcut(ages, q=self.n_age_bins, retbins=True, duplicates="drop")
            edges[0], edges[-1] = -np.inf, np.inf
            self.age_bin_edges_ = edges
        return self

    def transform(self, X):
        check_is_fitted(self, "age_bin_edges_")
        frame = X.copy()
        denominator = frame["age"].where(frame["age"] > 0)
        for output, source in (
            ("chol_per_age", "chol"),
            ("bps_per_age", "trestbps"),
            ("hr_ratio", "thalach"),
        ):
            frame[output] = (frame[source] / denominator).replace([np.inf, -np.inf], np.nan)
        frame["age_bin"] = pd.cut(
            frame["age"], bins=self.age_bin_edges_, labels=False, include_lowest=True
        ).astype(float)
        return frame

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        names = self.feature_names_in_ if input_features is None else input_features
        return np.asarray(
            list(names) + ["chol_per_age", "bps_per_age", "hr_ratio", "age_bin"], dtype=object
        )
