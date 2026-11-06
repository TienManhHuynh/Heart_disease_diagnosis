"""Bảy classifier; lưới nhỏ cho lần chạy sơ bộ, chưa phải tìm kiếm toàn diện."""

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

CORE_MODELS = (
    "naive_bayes",
    "knn",
    "decision_tree",
    "random_forest",
    "adaboost",
    "gradient_boosting",
    "xgboost",
)


def make_estimator(name: str, seed: int = 42):
    factories = {
        "naive_bayes": lambda: GaussianNB(),
        "knn": lambda: KNeighborsClassifier(),
        "decision_tree": lambda: DecisionTreeClassifier(random_state=seed),
        "random_forest": lambda: RandomForestClassifier(max_depth=5, random_state=seed, n_jobs=1),
        "adaboost": lambda: AdaBoostClassifier(random_state=seed),
        "gradient_boosting": lambda: GradientBoostingClassifier(max_depth=2, random_state=seed),
    }
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
    if name not in factories:
        raise ValueError(f"Model không hợp lệ: {name}")
    return factories[name]()


def parameter_grid(name: str) -> dict:
    grids = {
        "naive_bayes": {"model__var_smoothing": [1e-9, 1e-7]},
        "knn": {"model__n_neighbors": [3, 5, 9]},
        "decision_tree": {"model__max_depth": [3, 5, None]},
        "random_forest": {"model__n_estimators": [50, 100]},
        "adaboost": {"model__n_estimators": [50, 100]},
        "gradient_boosting": {"model__n_estimators": [50, 100]},
        "xgboost": {"model__n_estimators": [50, 100]},
    }
    if name not in grids:
        raise ValueError(f"Model không hợp lệ: {name}")
    return grids[name]
