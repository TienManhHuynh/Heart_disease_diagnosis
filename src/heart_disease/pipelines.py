"""Mọi bước học từ dữ liệu đều ở bên trong pipeline CV."""

from sklearn.pipeline import Pipeline

from .config import VARIANTS, Settings
from .features import AddClinicalFeatures
from .preprocessing import build_fe_preprocessor, build_original_preprocessor
from .selection import DecisionTreeSelector, MutualInformationSelector


def build_feature_pipeline(variant: str, settings: Settings):
    if variant not in VARIANTS:
        raise ValueError(f"Biến thể không hợp lệ: {variant}")
    if variant.startswith("fe"):
        steps = [
            ("add", AddClinicalFeatures(settings.age_bins)),
            ("preprocess", build_fe_preprocessor()),
            ("mi", MutualInformationSelector(settings.mi_k, settings.seed)),
        ]
    else:
        steps = [("preprocess", build_original_preprocessor())]
    if variant.endswith("_dt"):
        steps.append(("dt", DecisionTreeSelector(settings.dt_k, settings.seed)))
    return Pipeline(steps)


def build_model_pipeline(variant: str, estimator, settings: Settings):
    return Pipeline([("features", build_feature_pipeline(variant, settings)), ("model", estimator)])
