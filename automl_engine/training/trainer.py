from automl_engine.evaluation import scout_models
from automl_engine.reporting import print_section

from .workflow import execute_training_workflow
from .selection import resolve_best_model
from .finalizer import finalize_model


class ModelTrainer:

    def __init__(self, config, seed):
        self.config = config
        self.seed = seed

    def train(self, X, y, models, outer_cv, resolved):

        # ---------- Scout ----------
        if self.config.log:
            print_section("Global Pre-Screen")

        models, _ = scout_models(
            X, y,
            models,
            outer_cv,
            self.config,
        )

        # ---------- Evaluation ----------
        state, outer_scores = execute_training_workflow(
            X,
            y,
            models,
            outer_cv,
            self.config,
            resolved,
        )

        # ---------- Selection ----------
        best_model_name = resolve_best_model(state)

        # ---------- Final Fit ----------
        pipeline = finalize_model(
            best_model_name,
            state,
            models,
            X,
            y,
            self.config,
            self.seed,
        )

        return pipeline, state, outer_scores, best_model_name
