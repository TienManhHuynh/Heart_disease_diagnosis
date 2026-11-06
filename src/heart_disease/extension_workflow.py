"""Thí nghiệm bổ sung sau benchmark chính, không thay mô hình đã khóa."""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .artifacts import file_sha256, write_json
from .evaluation import classification_metrics, disease_probability, final_test_report
from .extension_models import build_extension_pipeline, extension_grid
from .workflows import MODELS, REPORTS, _load_verified


def run_extensions(settings):
    manifest, splits = _load_verified(settings)
    output = REPORTS / "extensions" / manifest["run_id"]
    complete = output / "manifest.json"
    if complete.exists():
        print("Dùng lại kết quả extensions đã hoàn tất.")
        return json.loads(complete.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.seed)
    X, y = splits["train"]
    X_val, y_val = splits["val"]
    rows, pipelines, records = [], {}, {}
    for model in settings.extension_models:
        for variant in settings.variants:
            key = f"{model}/{variant}"
            print(f"[extension {len(rows) + 1}] {key}", flush=True)
            search = GridSearchCV(
                build_extension_pipeline(model, variant, settings),
                extension_grid(model),
                cv=cv,
                scoring=settings.cv_scoring,
                n_jobs=1,
                error_score="raise",
                refit=True,
            ).fit(X, y)
            pipe = search.best_estimator_
            metrics = classification_metrics(
                y_val, pipe.predict(X_val), disease_probability(pipe, X_val)
            )
            rows.append(
                {
                    "model": model,
                    "variant": variant,
                    "cv_mean": float(search.best_score_),
                    "cv_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                    "best_params": search.best_params_,
                    **{
                        f"val_{name}": metrics[name]
                        for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
                    },
                }
            )
            directory = MODELS / manifest["run_id"] / "extensions" / model / variant
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "pipeline.joblib"
            joblib.dump(pipe, path)
            pd.DataFrame(search.cv_results_).to_csv(directory / "cv_results.csv", index=False)
            records[key] = {
                "pipeline": str(path.relative_to(REPORTS.parent)),
                "sha256": file_sha256(path),
            }
            pipelines[key] = pipe
    validation = pd.DataFrame(rows)
    # Chốt toàn bộ search trước khi tính test; không đưa extension vào selection/demo.
    validation.to_csv(output / "validation_results.csv", index=False)
    lock = {
        "parent_run_id": manifest["run_id"],
        "raw_sha256": manifest["raw_sha256"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": manifest["versions"],
        "core_selected_key_unchanged": manifest["selected_key"],
        "artifacts": records,
        "status": "exploratory_post_hoc",
        "note": "Added after core test results were observed; descriptive extension only, not an independent confirmatory comparison.",
    }
    write_json(output / "validation_lock.json", lock)
    test = final_test_report(pipelines, *splits["test"], output / "test")
    results = validation.merge(test, on=["model", "variant"], validate="one_to_one")
    results.to_csv(output / "results.csv", index=False)
    write_json(complete, lock)
    print(f"Đã hoàn tất {len(results)} cấu hình mở rộng; mô hình demo giữ nguyên.")
    return lock
