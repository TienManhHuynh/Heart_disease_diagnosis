"""Create a GitHub-ready source ZIP without local or generated runtime files."""

import json
from hashlib import sha256
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
REPO_NAME = "heart-disease-diagnosis-ml"
ROOT_FILES = (
    "README.md",
    "NOTICE.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "app.py",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
)
SOURCE_DIRECTORIES = (
    ".github",
    "configs",
    "data/raw",
    "docs",
    "notebooks",
    "reports",
    "scripts",
    "src",
    "tests",
)
ALLOWED_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".ipynb",
    ".csv",
    ".json",
    ".png",
    ".yml",
    ".yaml",
    ".txt",
}


def source_files(root=ROOT):
    files = [root / name for name in ROOT_FILES]
    for directory in SOURCE_DIRECTORIES:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or path.suffix not in ALLOWED_SUFFIXES:
                continue
            if any(
                part in {"__pycache__", ".ipynb_checkpoints", "executed_notebooks"}
                or part.endswith(".egg-info")
                for part in relative.parts
            ):
                continue
            if relative.as_posix() == "reports/delivery_verification.json":
                continue
            files.append(path)
    for name in (
        "data/processed/.gitkeep",
        "artifacts/models/.gitkeep",
        "artifacts/metadata/.gitkeep",
    ):
        path = root / name
        if path.exists():
            files.append(path)
    return sorted(set(files))


def main():
    destination = ROOT / "output/github"
    destination.mkdir(parents=True, exist_ok=True)
    files = source_files()
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("A required source file is missing.")
    manifest = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]
    archive_path = destination / f"{REPO_NAME}-source.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=f"{REPO_NAME}/{path.relative_to(ROOT).as_posix()}")
    with ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source ZIP integrity check failed.")
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    checksum = sha256(archive_path.read_bytes()).hexdigest()
    (destination / "SHA256.txt").write_text(f"{checksum}  {archive_path.name}\n", encoding="ascii")
    print(f"GitHub source: {len(files)} files, {archive_path.stat().st_size / 1024:.0f} KiB")
    print(archive_path)


if __name__ == "__main__":
    main()
