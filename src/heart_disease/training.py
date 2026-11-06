"""CV và validation dùng chung; không đọc dữ liệu test khi chọn cấu hình."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .config import Settings
from .evaluation import classification_metrics, disease_probability
from .models import make_estimator, parameter_grid
from .pipelines import build_model_pipeline


def tune_pipeline(pipeline, param_grid: dict, X_train, y_train, settings: Settings):
    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.seed)
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring=settings.cv_scoring,
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )
    return search.fit(X_train, y_train)


def benchmark_validation(splits: dict, settings: Settings) -> dict:
    rows, pipelines, searches = [], {}, {}
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    total = len(settings.models) * len(settings.variants)
    for model in settings.models:
        for variant in settings.variants:
            key = f"{model}/{variant}"
            print(f"[{len(rows) + 1}/{total}] CV: {key}", flush=True)
            pipeline = build_model_pipeline(variant, make_estimator(model, settings.seed), settings)
            search = tune_pipeline(pipeline, parameter_grid(model), X_train, y_train, settings)
            estimator = search.best_estimator_
            metrics = classification_metrics(
                y_val, estimator.predict(X_val), disease_probability(estimator, X_val)
            )
            index = search.best_index_
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "cv_scoring": settings.cv_scoring,
                    "cv_mean": float(search.best_score_),
                    "cv_std": float(search.cv_results_["std_test_score"][index]),
                    "best_params": search.best_params_,
                    **{
                        f"val_{k}": metrics[k]
                        for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
                    },
                }
            )
            pipelines[key], searches[key] = estimator, search
    return {"results": pd.DataFrame(rows), "pipelines": pipelines, "searches": searches}
