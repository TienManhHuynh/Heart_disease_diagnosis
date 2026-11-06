"""Kiểm tra rò rỉ fold, schema, tính bền vững và artifact của luồng ML."""

import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from heart_disease.artifacts import save_experiment
from heart_disease.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_settings
from heart_disease.data import load_cleveland, split_dataset
from heart_disease.evaluation import classification_metrics
from heart_disease.features import AddClinicalFeatures
from heart_disease.models import make_estimator
from heart_disease.pipelines import build_feature_pipeline, build_model_pipeline


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = load_settings()
        cls.raw = load_cleveland(cls.settings.raw_path)
        cls.splits = split_dataset(cls.raw, cls.settings)

    def test_splits_are_stratified_reproducible_and_disjoint(self):
        again = split_dataset(self.raw, self.settings)
        indices = []
        for name, expected_size in (("train", 242), ("val", 30), ("test", 31)):
            X, y = self.splits[name]
            self.assertEqual(len(X), expected_size)
            self.assertTrue(X.index.equals(y.index))
            self.assertTrue(X.index.equals(again[name][0].index))
            self.assertNotIn("target", X)
            self.assertLess(abs(y.mean() - self.raw.target.mean()), 0.03)
            indices.extend(X.index)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(set(indices), set(self.raw.index))

    def test_age_bins_are_not_refit_on_validation_and_zero_is_safe(self):
        X, _ = self.splits["train"]
        transformer = AddClinicalFeatures().fit(X)
        edges = transformer.age_bin_edges_.copy()
        probe = X.iloc[:3].copy()
        probe["age"] = [0, 1, 120]
        original = probe.copy(deep=True)
        result = transformer.transform(probe)
        np.testing.assert_array_equal(edges, transformer.age_bin_edges_)
        pd.testing.assert_frame_equal(probe, original)
        self.assertTrue(np.isnan(result.iloc[0]["chol_per_age"]))
        self.assertFalse(np.isinf(result.select_dtypes("number")).any().any())
        self.assertTrue(result.index.equals(probe.index))

    def test_all_variants_keep_schema_and_finite_values_for_unknown_categories(self):
        X, y = self.splits["train"]
        probe = self.splits["val"][0].copy()
        probe["cp"] = 99  # FE encoder phải xử lý nhóm chưa thấy ở train.
        for variant, count in (("original", 13), ("fe", 13), ("original_dt", 10), ("fe_dt", 10)):
            with self.subTest(variant=variant):
                pipeline = clone(build_feature_pipeline(variant, self.settings)).fit(X, y)
                train = pipeline.transform(X)
                validation = pipeline.transform(probe)
                self.assertEqual(train.shape, (242, count))
                self.assertEqual(validation.shape, (30, count))
                self.assertEqual(train.columns.tolist(), validation.columns.tolist())
                self.assertTrue(np.isfinite(validation.to_numpy()).all())
                self.assertTrue(validation.index.equals(probe.index))
                if variant == "original":
                    self.assertEqual(
                        train.columns.tolist(), list(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
                    )
                    np.testing.assert_allclose(train[list(NUMERIC_COLUMNS)].mean(), 0, atol=1e-12)

    def test_feature_fit_is_repeated_inside_each_cv_training_fold(self):
        X, y = self.splits["train"]
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        expected = [set(X.iloc[train].index) for train, _ in cv.split(X, y)]
        recorded = []
        original_fit = AddClinicalFeatures.fit

        def record_fit(instance, fold_X, fold_y=None):
            recorded.append(set(fold_X.index))
            return original_fit(instance, fold_X, fold_y)

        pipeline = build_model_pipeline("fe", make_estimator("naive_bayes"), self.settings)
        with patch.object(AddClinicalFeatures, "fit", record_fit):
            GridSearchCV(
                pipeline, {"model__var_smoothing": [1e-9]}, cv=cv, n_jobs=1, error_score="raise"
            ).fit(X, y)
        self.assertEqual(recorded[:2], expected)
        self.assertEqual(recorded[2], set(X.index))  # refit best trên toàn train.

    def test_artifact_roundtrip_keeps_predictions_and_probabilities(self):
        X, y = self.splits["train"]
        probe = self.splits["val"][0]
        pipeline = build_model_pipeline("fe_dt", make_estimator("naive_bayes"), self.settings).fit(
            X, y
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_experiment(pipeline, {"purpose": "test"}, directory)
            restored = joblib.load(paths["pipeline"])
            np.testing.assert_array_equal(pipeline.predict(probe), restored.predict(probe))
            np.testing.assert_allclose(pipeline.predict_proba(probe), restored.predict_proba(probe))

    def test_metrics_match_known_confusion_matrix_and_single_class_auc(self):
        metrics = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0], [0.9, 0.4, 0.6, 0.1])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        for name in ("accuracy", "precision", "recall", "f1"):
            self.assertEqual(metrics[name], 0.5)
        self.assertEqual(metrics["roc_auc"], 0.75)
        self.assertIsNone(classification_metrics([0, 0], [0, 0], [0.1, 0.2])["roc_auc"])


if __name__ == "__main__":
    unittest.main()
