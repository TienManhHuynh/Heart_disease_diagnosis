"""CLI cho kiểm tra dữ liệu, chuẩn bị và chạy thực nghiệm."""

import argparse
import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT, load_settings
from .data import inspect_raw_csv


def main(argv=None) -> int:
    # Windows có thể dùng code page không chứa tiếng Việt khi redirect output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Cleveland Heart Disease ML workflows")
    parser.add_argument(
        "command",
        choices=(
            "check-data",
            "plan",
            "prepare",
            "benchmark",
            "evaluate",
            "explain",
            "extensions",
            "report",
            "run",
        ),
    )
    parser.add_argument("--config", type=Path, help="Đường dẫn experiment.toml")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.command == "check-data":
            print(json.dumps(inspect_raw_csv(settings.raw_path), ensure_ascii=True, indent=2))
        elif args.command == "plan":
            print((PROJECT_ROOT / "docs" / "PROJECT_STATUS.md").read_text(encoding="utf-8"))
        elif args.command == "extensions":
            from .extension_workflow import run_extensions

            run_extensions(settings)
        elif args.command == "report":
            from .final_report import build_final_report

            build_final_report(settings)
        else:
            from . import workflows

            getattr(workflows, args.command)(settings)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Loi: {exc}\n")
    return 0
