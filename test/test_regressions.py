import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from automl_engine import AutoMLEngine, AutoMLConfig
from automl_engine.data.schema import infer_task
from automl_engine.evaluation.metrics import get_scorer_safe


class AutoMLRegressionTests(unittest.TestCase):
    def test_fast_and_thorough_presets_set_scout_folds(self):
        self.assertEqual(AutoMLConfig.fast().search.scout_folds, 2)
        self.assertEqual(AutoMLConfig.thorough().search.scout_folds, 5)

    def test_float_encoded_classes_are_classification(self):
        y = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0])
        self.assertEqual(infer_task(y), "classification")

    def test_continuous_float_target_is_regression(self):
        y = pd.Series(np.linspace(0.1, 9.7, 100))
        self.assertEqual(infer_task(y), "regression")

    def test_regression_metric_aliases_resolve(self):
        for metric in ("mse", "rmse", "mae", "r2"):
            self.assertTrue(callable(get_scorer_safe(metric)))

    def test_standard_cv_returns_fitted_predictable_pipeline(self):
        X, y = make_classification(
            n_samples=80,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        y = pd.Series(y, name="target")

        config = AutoMLConfig.fast()
        config.runtime.log = False
        config.models.include_models = ["dummy"]
        config.sampling.method = "none"
        config.feature_generation.method = "none"
        config.preprocessing.feature_selection_method = "none"
        config.optuna.enabled = False
        config.generate_optuna_plots = False

        engine = AutoMLEngine(config).fit((X, y))
        predictions = engine.predict(X.iloc[:5].copy())

        self.assertEqual(len(predictions), 5)
        self.assertEqual(engine.best_model_name_, "dummy")
        self.assertIsInstance(engine.best_score_, float)

    def test_regression_mae_runs_through_engine(self):
        X, y = make_regression(
            n_samples=80,
            n_features=4,
            noise=0.1,
            random_state=42,
        )
        X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        y = pd.Series(y, name="target")

        config = AutoMLConfig.fast()
        config.problem.task = "regression"
        config.problem.metric = "mae"
        config.runtime.log = False
        config.models.include_models = ["dummy"]
        config.feature_generation.method = "none"
        config.preprocessing.feature_selection_method = "none"
        config.optuna.enabled = False
        config.generate_optuna_plots = False

        engine = AutoMLEngine(config).fit((X, y))
        self.assertTrue(np.isfinite(engine.best_score_))


if __name__ == "__main__":
    unittest.main()
