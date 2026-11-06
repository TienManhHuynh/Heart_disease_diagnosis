"""Chuẩn bị -> CV/validation -> khóa cấu hình -> test -> SHAP."""

import json
import shutil
from datetime import datetime, timezone

import joblib
import pandas as pd

from .artifacts import (
    experiment_metadata,
    export_variants,
    file_sha256,
    save_experiment,
    settings_fingerprint,
    write_json,
)
from .config import PROJECT_ROOT, Settings
from .data import load_cleveland, split_dataset

METADATA = PROJECT_ROOT / "artifacts" / "metadata"
MODELS = PROJECT_ROOT / "artifacts" / "models"
REPORTS = PROJECT_ROOT / "reports"


def _load_verified(settings):
    path = METADATA / "benchmark.json"
    if not path.exists():
        raise FileNotFoundError("Chạy benchmark trước bước này.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["raw_sha256"] != file_sha256(settings.raw_path):
        raise ValueError("Dữ liệu thay đổi so với benchmark; chạy lại benchmark.")
    if manifest["settings_sha256"] != settings_fingerprint(settings):
        raise ValueError("Cấu hình thay đổi so với benchmark; chạy lại benchmark.")
    raw = load_cleveland(settings.raw_path)
    splits = {}
    for name, indices in manifest["split_indices"].items():
        subset = raw.loc[indices]
        splits[name] = (subset.drop(columns=[settings.target]), subset[settings.target])
    for name, record in manifest["artifacts"].items():
        if file_sha256(PROJECT_ROOT / record["pipeline"]) != record["sha256"]:
            raise ValueError(f"Artifact thay đổi: {name}")
    return manifest, splits


def _pipeline(manifest, key):
    return joblib.load(PROJECT_ROOT / manifest["artifacts"][key]["pipeline"])


def prepare(settings: Settings):
    from .eda import plot_eda, summarize

    raw = load_cleveland(settings.raw_path)
    splits = split_dataset(raw, settings)
    write_json(METADATA / "splits.json", experiment_metadata(settings, splits))
    eda = summarize(raw)
    output = REPORTS / "eda"
    output.mkdir(parents=True, exist_ok=True)
    for name, value in eda.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output / f"{name}.csv")
    write_json(output / "overview.json", eda["overview"])
    X_train, y_train = splits["train"]
    plot_eda(X_train.assign(target=y_train), REPORTS / "figures")
    exported = export_variants(splits, settings, PROJECT_ROOT / "data" / "processed")
    print(
        f"Đã xuất {len(exported['csv_paths'])} CSV và dataset.zip; EDA plot dùng train.", flush=True
    )
    return splits


def benchmark(settings: Settings):
    from .training import benchmark_validation

    raw = load_cleveland(settings.raw_path)
    splits = split_dataset(raw, settings)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = {
        **experiment_metadata(settings, splits),
        "run_id": run_id,
        "selection_rule": "Highest train CV mean; ties by validation F1, then model and variant names.",
        "artifacts": {},
    }
    result = benchmark_validation(splits, settings)
    table = result["results"]
    for key, pipeline in result["pipelines"].items():
        model, variant = key.split("/")
        record = table[(table.model == model) & (table.variant == variant)].iloc[0].to_dict()
        paths = save_experiment(
            pipeline, {**manifest, "result": record}, MODELS / run_id / model / variant
        )
        manifest["artifacts"][key] = {
            "pipeline": str(paths["pipeline"].relative_to(PROJECT_ROOT)),
            "metadata": str(paths["metadata"].relative_to(PROJECT_ROOT)),
            "sha256": paths["sha256"],
        }
        pd.DataFrame(result["searches"][key].cv_results_).to_csv(
            paths["metadata"].parent / "cv_results.csv", index=False
        )
    ranked = table.sort_values(
        ["cv_mean", "val_f1", "model", "variant"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    selected = ranked.iloc[0]
    manifest["selected_key"] = f"{selected['model']}/{selected['variant']}"
    table.insert(0, "run_id", run_id)
    table.insert(1, "seed", settings.seed)
    table.to_csv(REPORTS / "validation_results.csv", index=False)
    write_json(METADATA / "benchmark.json", manifest)
    selected_path = PROJECT_ROOT / manifest["artifacts"][manifest["selected_key"]]["pipeline"]
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_path, MODELS / "selected_pipeline.joblib")
    write_json(
        METADATA / "selected_model.json",
        {
            "run_id": run_id,
            "selected_key": manifest["selected_key"],
            "selection_rule": manifest["selection_rule"],
            "raw_sha256": manifest["raw_sha256"],
            "pipeline_sha256": file_sha256(MODELS / "selected_pipeline.joblib"),
            "input_columns": list(splits["train"][0].columns),
            "versions": manifest["versions"],
        },
    )
    print(
        f"Đã khóa lựa chọn theo CV/validation: {manifest['selected_key']}. Chưa dùng test.",
        flush=True,
    )
    return manifest


def evaluate(settings: Settings):
    from .evaluation import final_test_report, repeated_cv
    from .reporting import write_run_report

    manifest, splits = _load_verified(settings)
    output = REPORTS / "runs" / manifest["run_id"]
    complete = output / "evaluation.json"
    if complete.exists():
        print("Đã có báo cáo test cho run này; dùng lại kết quả, không đánh giá lại.")
        return json.loads(complete.read_text(encoding="utf-8"))
    validation = pd.read_csv(REPORTS / "validation_results.csv")
    if set(validation.run_id) != {manifest["run_id"]}:
        raise ValueError("Validation report không thuộc run hiện tại.")
    pipelines = {key: _pipeline(manifest, key) for key in manifest["artifacts"]}
    X_test, y_test = splits["test"]
    table = final_test_report(
        pipelines, X_test, y_test, output, plot_keys=[manifest["selected_key"]]
    )
    combined = validation.merge(table, on=["model", "variant"], validate="one_to_one")
    combined.to_csv(REPORTS / "results.csv", index=False)
    X_train, y_train = splits["train"]
    stability = repeated_cv(
        pipelines[manifest["selected_key"]],
        X_train,
        y_train,
        seed=settings.seed,
        n_splits=settings.cv_folds,
    )
    result = {
        "run_id": manifest["run_id"],
        "selected_key": manifest["selected_key"],
        "test_samples": len(X_test),
        "stability_training_only": stability,
        "test_report_directory": str(output.relative_to(PROJECT_ROOT)),
    }
    write_json(complete, result)
    write_run_report(manifest, combined, result, REPORTS / "baseline_report.md")
    print(f"Đã báo cáo test {len(X_test)} mẫu cho {len(table)} cấu hình đã khóa.", flush=True)
    return result


def explain(settings: Settings):
    from .explain import explain_tree_pipeline

    manifest, splits = _load_verified(settings)
    key = "random_forest/original"
    if key not in manifest["artifacts"]:
        raise ValueError("SHAP sơ bộ yêu cầu random_forest và original trong cấu hình.")
    X_test, y_test = splits["test"]
    result = explain_tree_pipeline(
        _pipeline(manifest, key),
        X_test,
        REPORTS / "runs" / manifest["run_id"] / "shap",
        y_true=y_test,
    )
    result.update(run_id=manifest["run_id"], model_key=key)
    write_json(METADATA / "shap.json", result)
    print(
        f"Đã tạo SHAP cho {key}; sai số additivity: {result['max_additivity_error']:.2e}.",
        flush=True,
    )
    return result


def run(settings: Settings):
    prepare(settings)
    benchmark(settings)
    evaluate(settings)
    explain(settings)
