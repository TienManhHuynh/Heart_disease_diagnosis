"""CLI phải hiển thị tiếng Việt cả khi Windows mặc định dùng encoding cp1252."""

import os
import subprocess
import sys
import unittest

from heart_disease.config import PROJECT_ROOT


class CliTests(unittest.TestCase):
    def test_plan_survives_legacy_windows_encoding(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        environment["PYTHONIOENCODING"] = "cp1252"
        result = subprocess.run(
            [sys.executable, "-m", "heart_disease", "plan"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        output = result.stdout.decode("utf-8")
        self.assertIn("Checklist", output)
        self.assertIn("triển khai", output)
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
