"""Kiểm tra ánh xạ cluster theo train và preprocessing bên trong OOF stacking."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone

from heart_disease.config import load_settings
from heart_disease.data import load_cleveland, split_dataset
from heart_disease.extension_models import KMeansClassifier, build_stacking_baseline
from heart_disease.features import AddClinicalFeatures


class ExtensionTests(unittest.TestCase):
    def test_cluster_mapping_uses_training_labels_and_returns_valid_probability(self):
        X = np.array([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]])
        y = np.array([1, 1, 1, 0, 0, 0])
        estimator = clone(KMeansClassifier()).fit(X, y)
        np.testing.assert_array_equal(estimator.predict(X), y)
        np.testing.assert_allclose(estimator.predict_proba(X).sum(axis=1), 1)
        reverse = clone(estimator).fit(X, 1 - y)
        np.testing.assert_array_equal(reverse.predict(X), 1 - y)
        np.testing.assert_array_equal(estimator.predict(X), y)

    def test_stacking_fits_feature_engineering_separately_in_oof_folds(self):
        settings = load_settings()
        X, y = split_dataset(load_cleveland(settings.raw_path), settings)["train"]
        X, y = X.iloc[:96], y.iloc[:96]
        sizes = []
        original_fit = AddClinicalFeatures.fit

        def recording_fit(instance, values, labels=None):
            sizes.append(len(values))
            return original_fit(instance, values, labels)

        estimator = build_stacking_baseline(42, "fe", settings)
        with patch.object(AddClinicalFeatures, "fit", recording_fit):
            estimator.fit(X, y)
        self.assertEqual(sizes.count(96), 3)
        self.assertEqual(sizes.count(64), 9)
        np.testing.assert_allclose(estimator.predict_proba(X).sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
