"""Feature selectors có trạng thái fit riêng cho từng training fold."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from .config import CATEGORICAL_COLUMNS


class _ColumnSelector(TransformerMixin, BaseEstimator):
    def transform(self, X):
        check_is_fitted(self, "selected_columns_")
        return X.loc[:, self.selected_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_columns_")
        return np.asarray(self.selected_columns_, dtype=object)

    def _select(self, scores):
        if self.k < 1 or scores.empty:
            raise ValueError("Cần k >= 1 và ít nhất một đặc trưng hợp lệ.")
        self.scores_ = scores.sort_values(ascending=False, kind="stable")
        self.selected_columns_ = self.scores_.head(self.k).index.tolist()
        return self


class MutualInformationSelector(_ColumnSelector):
    def __init__(self, k=13, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.nonconstant_columns_ = X.columns[X.nunique(dropna=False) > 1].tolist()
        if not self.nonconstant_columns_:
            raise ValueError("Không có cột biến thiên để tính MI.")
        subset = X[self.nonconstant_columns_]
        # FE preprocessor đặt tên one-hot bằng tên cột categorical + '_'.
        prefixes = tuple(f"{name}_" for name in (*CATEGORICAL_COLUMNS, "age_bin"))
        self.discrete_mask_ = np.array([str(c).startswith(prefixes) for c in subset.columns])
        scores = mutual_info_classif(
            subset, y, discrete_features=self.discrete_mask_, random_state=self.random_state
        )
        return self._select(pd.Series(scores, index=subset.columns))


class DecisionTreeSelector(_ColumnSelector):
    def __init__(self, k=10, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.estimator_ = DecisionTreeClassifier(random_state=self.random_state).fit(X, y)
        self.importance_ = pd.Series(self.estimator_.feature_importances_, index=X.columns)
        return self._select(self.importance_)
