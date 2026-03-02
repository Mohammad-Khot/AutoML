# automl_engine/training/trainer.py

from automl_engine.core import MODEL_PRIORITY, select_best_model
from automl_engine.preprocessing import build_pipeline
from automl_engine.evaluation import evaluate_models, scout_models
from automl_engine.orchestration.nested import nested_cv
from automl_engine.utils.console import print_section


class ModelTrainer:

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def train(self, X, y, models, outer_cv, resolved):

        # ---------- Scout ----------
        if self.config.log:
            print_section("Global Pre-Screen")
        models, _ = scout_models(X, y, models, outer_cv, self.config)

        # ---------- Evaluation ----------
        if not self.config.nested_cv:
            if self.config.log:
                print_section("Standard Cross Validation")
            state = evaluate_models(X, y, models, outer_cv, self.config, resolved, "OUTER_CV")
            outer_scores = state.scores
        else:
            if self.config.log:
                print_section("Nested Evaluation")
            outer_result = nested_cv(X, y, models, outer_cv, self.config, resolved)
            outer_scores = getattr(outer_result, "scores", outer_result)

            if self.config.log:
                print_section("Final Model Fit")
            state = evaluate_models(X, y, models, outer_cv, self.config, resolved, "FINAL_FIT")

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

        if self.config.log:
            print_section("Deployment Model Ready")
        best_pipeline.fit(X, y)

        return best_pipeline, state, outer_scores, best_model_name
