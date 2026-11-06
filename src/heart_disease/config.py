"""Schema dữ liệu và cấu hình dùng chung, không phụ thuộc thư viện ML."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
)
NUMERIC_COLUMNS = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORICAL_COLUMNS = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
VARIANTS = ("original", "fe", "original_dt", "fe_dt")


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    data_path: str = "data/raw/cleveland.csv"
    target: str = "target"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    cv_folds: int = 5
    cv_scoring: str = "accuracy"
    mi_k: int = 13
    dt_k: int = 10
    age_bins: int = 5
    variants: tuple[str, ...] = VARIANTS
    models: tuple[str, ...] = (
        "naive_bayes",
        "knn",
        "decision_tree",
        "random_forest",
        "adaboost",
        "gradient_boosting",
        "xgboost",
    )
    extension_models: tuple[str, ...] = ("kmeans", "stacking")

    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / self.data_path


def load_settings(path: Path | None = None) -> Settings:
    path = path or PROJECT_ROOT / "configs" / "experiment.toml"
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    for key in ("variants", "models", "extension_models"):
        if key in values:
            values[key] = tuple(values[key])
    settings = Settings(**values)
    ratios = (settings.train_ratio, settings.val_ratio, settings.test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Tỉ lệ train/val/test phải dương và có tổng bằng 1.")
    if settings.target != COLUMNS[-1]:
        raise ValueError("Schema Cleveland trong project dùng target='target'.")
    if set(settings.variants) - set(VARIANTS):
        raise ValueError("Tên biến thể không được hỗ trợ.")
    if settings.cv_folds < 2 or min(settings.mi_k, settings.dt_k, settings.age_bins) < 1:
        raise ValueError("cv_folds >= 2; số đặc trưng và số nhóm tuổi phải >= 1.")
    return settings
