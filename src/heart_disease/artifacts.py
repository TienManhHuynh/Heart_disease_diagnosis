"""Lưu pipeline đầy đủ và dấu vết thực nghiệm; không tải artifact từ mạng."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def file_sha256(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def settings_fingerprint(settings):
    return sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()


def library_versions():
    result = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "streamlit", "joblib"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def experiment_metadata(settings, splits):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": asdict(settings),
        "settings_sha256": settings_fingerprint(settings),
        "raw_sha256": file_sha256(settings.raw_path),
        "versions": library_versions(),
        "split_indices": {name: X.index.tolist() for name, (X, _) in splits.items()},
    }


def save_experiment(fitted_pipeline, metadata: dict, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pipeline.joblib"
    joblib.dump(fitted_pipeline, model_path)
    metadata = {
        **metadata,
        "pipeline_sha256": file_sha256(model_path),
        "input_columns": fitted_pipeline.feature_names_in_.tolist(),
        "output_columns": fitted_pipeline.named_steps["features"].get_feature_names_out().tolist(),
    }
    metadata_path = write_json(output_dir / "metadata.json", metadata)
    return {
        "pipeline": model_path,
        "metadata": metadata_path,
        "sha256": metadata["pipeline_sha256"],
    }


def export_variants(splits: dict, settings, output_dir):
    from .pipelines import build_feature_pipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefixes = {"original": "raw", "fe": "fe", "original_dt": "dt", "fe_dt": "fe_dt"}
    paths, information = [], {}
    for variant in settings.variants:
        pipe = build_feature_pipeline(variant, settings)
        X_train, y_train = splits["train"]
        pipe.fit(X_train, y_train)
        for split_name, (X, y) in splits.items():
            transformed = pipe.transform(X)
            if not transformed.index.equals(y.index):
                raise ValueError("Feature/target index mismatch.")
            output = transformed.copy()
            output[settings.target] = y
            path = output_dir / f"{prefixes[variant]}_{split_name}.csv"
            output.to_csv(path, index=False)
            paths.append(path)
        information[variant] = {"columns": pipe.get_feature_names_out().tolist()}
        for name in ("mi", "dt"):
            if name in pipe.named_steps:
                information[variant][f"{name}_scores"] = pipe.named_steps[name].scores_.to_dict()
    write_json(output_dir / "feature_manifest.json", information)
    archive = output_dir / "dataset.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as stream:
        for path in paths:
            stream.write(path, arcname=f"splits/{path.name}")
    return {"csv_paths": paths, "archive": archive, "features": information}
