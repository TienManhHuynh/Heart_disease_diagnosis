"""Nạp artifact local và dự đoán một hàng với cùng schema lúc huấn luyện."""

import json

import joblib
import pandas as pd

from .artifacts import file_sha256
from .config import COLUMNS, PROJECT_ROOT
from .evaluation import disease_probability


def load_selected_pipeline():
    artifact = PROJECT_ROOT / "artifacts" / "models" / "selected_pipeline.joblib"
    metadata = PROJECT_ROOT / "artifacts" / "metadata" / "selected_model.json"
    if not artifact.exists() or not metadata.exists():
        raise FileNotFoundError("Chạy python -m heart_disease benchmark trước khi mở demo.")
    information = json.loads(metadata.read_text(encoding="utf-8"))
    if file_sha256(artifact) != information["pipeline_sha256"]:
        raise ValueError("Pipeline không khớp metadata; chạy lại benchmark.")
    return joblib.load(artifact), information


def predict_patient(pipeline, patient: dict):
    expected = list(COLUMNS[:-1])
    if set(patient) != set(expected):
        raise ValueError("Đầu vào phải gồm đúng 13 cột Cleveland, không có target.")
    frame = pd.DataFrame([patient], columns=expected).apply(pd.to_numeric, errors="raise")
    return {
        "label": int(pipeline.predict(frame)[0]),
        "disease_probability": float(disease_probability(pipeline, frame)[0]),
    }
