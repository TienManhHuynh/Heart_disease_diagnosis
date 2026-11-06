"""Đóng gói các file có ích; loại .venv, cache, file tạm và ZIP đệ quy."""

import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "output/delivery"
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for directory in (
        ".github",
        "configs",
        "data",
        "src",
        "notebooks",
        "artifacts",
        "reports",
        "docs",
        "scripts",
        "tests",
        "output/pdf",
    ):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and not any(
                p in {"__pycache__", ".ipynb_checkpoints"} or p.endswith(".egg-info")
                for p in path.parts
            ):
                if path.suffix not in {".pyc", ".pyo"}:
                    files.append(path)
    for name in (
        "README.md",
        "NOTICE.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-lock.txt",
        "app.py",
        ".gitignore",
        ".gitattributes",
        ".editorconfig",
    ):
        files.append(ROOT / name)
    files = sorted(set(files))
    manifest = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]
    (destination / "file_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    path = destination / "Heart_Disease_Project_Complete.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(
                file, arcname="heart_disease_project/" + file.relative_to(ROOT).as_posix()
            )
        archive.write(
            destination / "file_manifest.json", arcname="heart_disease_project/file_manifest.json"
        )
    checksum = sha256(path.read_bytes()).hexdigest()
    (destination / "SHA256.txt").write_text(f"{checksum}  {path.name}\n", encoding="ascii")
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
    print(f"Packaged {len(files)} files; ZIP integrity passed: {path}")


if __name__ == "__main__":
    main()
