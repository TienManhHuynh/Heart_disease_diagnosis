"""Kiểm tra phần đã triển khai: dữ liệu hợp lệ và lỗi nhãn không bị bỏ qua."""

import csv
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from heart_disease.config import load_settings
from heart_disease.data import inspect_raw_csv


class DataContractTests(unittest.TestCase):
    def inspect_rows(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(rows)
            return inspect_raw_csv(path)

    def test_supplied_dataset_matches_expected_schema(self):
        summary = inspect_raw_csv(load_settings().raw_path)
        self.assertEqual((summary["rows"], summary["columns"]), (303, 14))
        self.assertEqual(summary["binary_target_counts"], {0: 164, 1: 139})
        self.assertEqual(summary["missing_by_column"]["ca"], 4)
        self.assertEqual(summary["missing_by_column"]["thal"], 2)

    def test_dataset_matches_reference_checksum(self):
        self.assertEqual(
            sha256(load_settings().raw_path.read_bytes()).hexdigest(),
            "70a023033aaf840596737ad250eea055762ad825036ac73b5b29168dc638c7c2",
        )

    def test_missing_features_are_reported_not_dropped(self):
        row = ["1"] * 14
        row[11], row[12] = "?", ""
        summary = self.inspect_rows([row])
        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["missing_by_column"]["ca"], 1)
        self.assertEqual(summary["missing_by_column"]["thal"], 1)

    def test_missing_or_invalid_target_is_rejected_before_binarization(self):
        for target in ("?", "", "-1", "5", "0.5", "NaN"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.inspect_rows([["1"] * 13 + [target]])

    def test_wrong_width_or_nonfinite_feature_is_rejected(self):
        for row in (["1"] * 13, ["inf"] + ["1"] * 13, ["NaN"] + ["1"] * 13):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.inspect_rows([row])

    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            self.inspect_rows([])


if __name__ == "__main__":
    unittest.main()
