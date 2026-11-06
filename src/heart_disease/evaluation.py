"""Đánh giá validation, holdout cuối và CV độ ổn định của cấu hình cố định."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from .artifacts import write_json


def disease_probability(pipeline, X):
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Estimator không có lớp 1.")
    return pipeline.predict_proba(X)[:, classes.index(1)]


def classification_metrics(y_true, y_pred, y_probability=None) -> dict:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_probability is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_probability))
    else:
        result["roc_auc_note"] = "Unavailable: probability missing or y_true has one class."
    return result


def final_test_report(fitted_pipelines: dict, X_test, y_test, output_dir, plot_keys=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    for key, pipeline in fitted_pipelines.items():
        model, variant = key.split("/", 1)
        prediction = pipeline.predict(X_test)
        metrics = classification_metrics(y_test, prediction, disease_probability(pipeline, X_test))
        rows.append(
            {
                "model": model,
                "variant": variant,
                **{
                    f"test_{k}": metrics[k]
                    for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
                },
            }
        )
        details[key] = {
            "metrics": metrics,
            "classification_report": classification_report(
                y_test,
                prediction,
                labels=[0, 1],
                target_names=["no_disease", "disease"],
                output_dict=True,
                zero_division=0,
            ),
        }
        if key in (plot_keys or []):
            display = ConfusionMatrixDisplay.from_predictions(
                y_test,
                prediction,
                labels=[0, 1],
                display_labels=["No disease", "Disease"],
                cmap="Blues",
                colorbar=False,
            )
            display.ax_.set_title(key)
            display.figure_.savefig(
                output_dir / f"confusion_{model}_{variant}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(display.figure_)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "test_metrics.csv", index=False)
    write_json(output_dir / "classification_reports.json", details)
    return table


def repeated_cv(pipeline, X, y, seed=42, n_splits=5, n_repeats=3) -> dict:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=1, error_score="raise")
    return {
        "scoring": "f1",
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "scores": scores.tolist(),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "note": "Fixed hyperparameters chosen using this training subset; descriptive stability check, not nested unbiased model selection.",
    }
