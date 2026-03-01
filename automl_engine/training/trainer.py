# automl_engine/training/trainer.py
import time

from automl_engine.core import MODEL_PRIORITY, select_best_model
from automl_engine.preprocessing import build_pipeline
from automl_engine.evaluation import evaluate_models, scout_models
from automl_engine.evaluation.nested import nested_cv


class ModelTrainer:

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def train(self, X, y, models, outer_cv, resolved):

        # ---------- Scout ----------
        print("\n=== GLOBAL PRE-SCREEN ===")
        models, _ = scout_models(X, y, models, outer_cv, self.config)

        # ---------- Evaluation ----------
        if not self.config.nested_cv:
            print("\n=== RUNNING STANDARD CROSS_EVALUATION ===")
            state = evaluate_models(X, y, models, outer_cv, self.config, resolved)
            outer_scores = state.scores
        else:
            print("\n=== RUNNING NESTED EVALUATION ===")
            outer_result = nested_cv(X, y, models, outer_cv, self.config, resolved)
            outer_scores = getattr(outer_result, "scores", outer_result)

            print("\n=== FINAL MODEL SELECTION ===")
            state = evaluate_models(X, y, models, outer_cv, self.config, resolved)

        if not state.scores:
            raise RuntimeError("No models successfully evaluated.")

        # ---------- Final Selection ----------
        best_model_name = select_best_model(state.scores, MODEL_PRIORITY)
        best_info = models[best_model_name]

        best_pipeline = state.get_pipeline(best_model_name)

        if best_pipeline is None:
            best_pipeline = build_pipeline(
                best_info, X, self.config, seed=self.seed
            )

        best_pipeline.fit(X, y)

        return best_pipeline, state, outer_scores, best_model_name
