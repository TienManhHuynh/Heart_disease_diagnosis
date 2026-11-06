"""Kiểm tra artifact/báo cáo hiện hành trước khi đóng gói, không train lại."""

import ast
import json
import sys
import unittest
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from heart_disease.artifacts import file_sha256, write_json
from heart_disease.config import load_settings
from heart_disease.inference import load_selected_pipeline
from heart_disease.workflows import _load_verified


def main():
    source_paths = [ROOT / "app.py"]
    for directory in ("src", "scripts", "tests"):
        source_paths.extend((ROOT / directory).rglob("*.py"))
    for source in source_paths:
        ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test_result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not test_result.wasSuccessful():
        raise RuntimeError("Unit tests failed.")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception and len(app.number_input) == 5 and len(app.selectbox) == 8
    app.button[0].click().run()
    assert not app.exception and len(app.metric) == 1
    manifest, _ = _load_verified(load_settings())
    load_selected_pipeline()  # Kiểm tra hash của bản dùng trong demo.
    core = pd.read_csv(ROOT / "reports/results.csv")
    extension = pd.read_csv(ROOT / "reports/extensions" / manifest["run_id"] / "results.csv")
    assert len(core) == 28 and len(extension) == 8
    assert len(core[["model", "variant"]].drop_duplicates()) == 28
    assert len(extension[["model", "variant"]].drop_duplicates()) == 8
    for frame in (core, extension):
        score_columns = [
            c
            for c in frame
            if c.startswith(("cv_", "val_", "test_")) and pd.api.types.is_numeric_dtype(frame[c])
        ]
        assert np.isfinite(frame[score_columns].to_numpy()).all()
    assert (ROOT / "output/pdf/Heart_Disease_Project_Report.pdf").read_bytes().startswith(b"%PDF")
    for name in ("final_report.md",):
        text = (ROOT / "reports" / name).read_text(encoding="utf-8")
        assert "TODO" not in text and "NotImplemented" not in text
    for name in ("00_eda", "01_feature_variants", "02_benchmark", "03_treeshap"):
        path = ROOT / "reports/executed_notebooks" / f"{name}.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert all(c["execution_count"] is not None for c in code_cells)
        assert not any(o["output_type"] == "error" for c in code_cells for o in c["outputs"])
    with ZipFile(ROOT / "data/processed/dataset.zip") as archive:
        assert len(archive.namelist()) == 12 and archive.testzip() is None
    errors = pd.read_csv(ROOT / "reports/error_analysis.csv")
    assert errors.error_type.value_counts().to_dict() == {"FP": 2, "FN": 1}
    result = {
        "run_id": manifest["run_id"],
        "core_configurations": 28,
        "extension_configurations": 8,
        "executed_notebooks_without_errors": 4,
        "unit_tests_passed": test_result.testsRun,
        "demo_apptest_passed": True,
        "pdf_header_and_report_checks_passed": True,
        "selected_pipeline_sha256": file_sha256(ROOT / "artifacts/models/selected_pipeline.joblib"),
    }
    write_json(ROOT / "reports/delivery_verification.json", result)
    print(
        "Delivery verified: 36 configurations, 4 executed notebooks, reports and artifacts consistent."
    )


if __name__ == "__main__":
    main()
