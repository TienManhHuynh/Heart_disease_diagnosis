"""Chạy bằng Python trong .venv đã cài dependencies: python scripts/check_project.py."""

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_disease.config import load_settings
from heart_disease.data import inspect_raw_csv


def main() -> int:
    paths = [ROOT / "app.py"]
    for directory in ("src", "scripts", "tests"):
        paths.extend(sorted((ROOT / directory).rglob("*.py")))
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    print(f"[OK] Syntax: {len(paths)} Python files.", flush=True)

    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if len(notebook_paths) != 4:
        raise ValueError("Expected four learning notebooks.")
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4 or not notebook.get("cells"):
            raise ValueError(f"Invalid notebook structure: {path}")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=path.name)
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    raise ValueError(f"Expected unexecuted template cells: {path}")
    print(f"[OK] Notebook JSON and code syntax: {len(notebook_paths)} templates.", flush=True)

    settings = load_settings()
    summary = inspect_raw_csv(settings.raw_path)
    print(
        f"[OK] Config and raw CSV: {summary['rows']} rows, {summary['columns']} columns.",
        flush=True,
    )
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    if not result.wasSuccessful():
        return 1
    print(
        "[OK] Source, notebook syntax, data contract and pipeline unit tests passed. Full workflow outputs are verified separately."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
