"""KMeans có ánh xạ nhãn và stacking với preprocessing trong từng base learner."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .config import Settings
from .models import make_estimator
from .pipelines import build_model_pipeline


class KMeansClassifier(ClassifierMixin, BaseEstimator):
    """Cluster-to-class chỉ học từ y_train; xác suất là tần suất có làm trơn.

    Xác suất này không phải xác suất từ mô hình phân loại được hiệu chỉnh.
    Nhãn cluster tùy ý nên cần ánh xạ theo nhãn của training subset hiện tại.
    """

    def __init__(self, n_clusters=2, n_init=10, random_state=42, smoothing=1.0):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.smoothing = smoothing

    def fit(self, X, y):
        if self.smoothing <= 0:
            raise ValueError("smoothing phải > 0.")
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        values, labels = check_X_y(X, y)
        self.n_features_in_ = values.shape[1]
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        self.clusterer_ = KMeans(
            n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state
        ).fit(values)
        prior = np.bincount(encoded, minlength=len(self.classes_)) / len(encoded)
        self.cluster_probabilities_ = np.empty((self.n_clusters, len(self.classes_)))
        for cluster in range(self.n_clusters):
            members = encoded[self.clusterer_.labels_ == cluster]
            counts = np.bincount(members, minlength=len(self.classes_))
            self.cluster_probabilities_[cluster] = (counts + self.smoothing * prior) / (
                len(members) + self.smoothing
            )
        self.cluster_to_class_ = self.classes_[self.cluster_probabilities_.argmax(axis=1)]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "cluster_probabilities_")
        values = check_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Số feature không khớp lúc fit.")
        return self.cluster_probabilities_[self.clusterer_.predict(values)]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def build_kmeans_baseline(seed=42):
    return KMeansClassifier(random_state=seed)


def build_stacking_baseline(seed=42, variant="original", settings=None):
    settings = settings or Settings(seed=seed)
    estimators = []
    for name in ("knn", "decision_tree", "naive_bayes"):
        estimator = make_estimator(name, seed)
        if name == "decision_tree":
            estimator.set_params(max_depth=5)
        estimators.append((name, build_model_pipeline(variant, estimator, settings)))
    # Pipeline riêng ở TỪNG base learner để preprocessing/MI/DT được fit lại
    # ngay trong các fold tạo dự đoán OOF của stacking.
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def build_extension_pipeline(name, variant, settings):
    if name == "kmeans":
        return build_model_pipeline(variant, build_kmeans_baseline(settings.seed), settings)
    if name == "stacking":
        return build_stacking_baseline(settings.seed, variant, settings)
    raise ValueError(f"Extension không hợp lệ: {name}")


def extension_grid(name):
    if name == "kmeans":
        return {"model__n_init": [10, 20]}
    if name == "stacking":
        return {"final_estimator__C": [0.1, 1.0]}
    raise ValueError(name)
