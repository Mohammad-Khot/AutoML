import warnings

import pandas as pd
from typing import cast

from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelEncoder

from automl_engine.data import (
    infer_task,
    apply_leakage_policy
)

from automl_engine.evaluation import (
    resolve_metric,
    get_cv_object,
)

from automl_engine.planning.models.registry import (
    MODEL_REGISTRY,
    COST_LOW,
    COST_MEDIUM,
)

from automl_engine.planning.models.selector import (
    is_model_suitable,
)

from automl_engine.planning.metadata import DataInfo

from automl_engine import AutoMLConfig
from automl_engine.planning.config import MetricName, SearchConfig, ModelSpaceConfig
from automl_engine.planning.models.spec import ModelSpec
from automl_engine.planning.experiment.resolved import (
    ResolvedConfig,
    ResolvedProblemConfig,
    ResolvedCVConfig,
    ResolvedPreprocessingConfig,
    ResolvedDimensionalityReductionConfig,
    ResolvedModelSpaceConfig,
    ResolvedFeatureGenerationConfig,
    ResolvedSearchConfig,
    ResolvedDataQualityConfig,
    ResolvedRuntimeConfig,
    ResolvedOptunaConfig,
    ResolvedArtifactsConfig,
)
from automl_engine.planning.sampler import resolve_sampling


class ExperimentResolver:

    def __init__(self, config: AutoMLConfig) -> None:
        self.config = config

    def resolve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, ResolvedConfig]:

        X = X.copy()
        y = y.copy()

        # ───────── Leakage Detection ─────────

        X, leaks = apply_leakage_policy(
            X,
            y,
            dq_config=self.config.data_quality,
        )

        # ───────── Task Inference ─────────
        if self.config.problem.task is not None:
            inferred = infer_task(y)

            if inferred != self.config.problem.task:
                warnings.warn(
                    f"User-defined task '{self.config.problem.task}' "
                    f"conflicts with inferred task '{inferred}'. Using user-defined task."
                )

            task = self.config.problem.task

        else:
            task = infer_task(y)

        # ───────── Label Encoding ─────────
        label_encoder: LabelEncoder | None = None
        if task == "classification" and not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = pd.Series(
                label_encoder.fit_transform(y),
                index=y.index,
            )

        # ───────── Metric Resolution ─────────
        metric: MetricName = resolve_metric(task, self.config.problem.metric)

        # ───────── Data Metadata ─────────
        data_info: DataInfo = DataInfo.from_data(X, y)

        # ───────── Sampling Resolution ─────────
        sampling_config = resolve_sampling(self.config, data_info, task)

        # ───────── Model Filtering ─────────
        models: dict[str, ModelSpec] = self._filter_models(
            dict(MODEL_REGISTRY[task]),
            data_info,
            self.config,
        )

        # ───────── Phase 1: Build Resolved Config (without CV object) ─────────
        resolved = ResolvedConfig(

            problem=ResolvedProblemConfig(
                task=task,
                metric=metric,
                target=self.config.problem.target or "",
            ),

            cv=ResolvedCVConfig(
                folds=self.config.cv.folds,
                strategy=self.config.cv.strategy,
                use_nested_cv=self.config.cv.use_nested_cv,
                repeats=self.config.cv.repeats,
            ),

            preprocessing=ResolvedPreprocessingConfig(
                scaling_mode=self.config.preprocessing.scaling_mode,
                scaler_kind=self.config.preprocessing.scaler_kind,
                encoding_strategy=self.config.preprocessing.encoding_strategy,
                max_cardinality_one_hot=self.config.preprocessing.max_cardinality_one_hot,
                feature_selection_method=self.config.preprocessing.feature_selection_method,
                imputation_strategy=self.config.preprocessing.imputation_strategy,
                add_missing_indicator=self.config.preprocessing.add_missing_indicator,
            ),

            feature_generation=ResolvedFeatureGenerationConfig(
                method=self.config.feature_generation.method,
                strategy=self.config.feature_generation.strategy,
                max_polynomial_degree=self.config.feature_generation.max_polynomial_degree,
                interaction_only=self.config.feature_generation.interaction_only,
                max_generated_features=self.config.feature_generation.max_generated_features,
                subsample_ratio=self.config.feature_generation.subsample_ratio,
            ),

            dimensionality_reduction=ResolvedDimensionalityReductionConfig(
                method=self.config.dimensionality_reduction.method,
                n_components=self.config.dimensionality_reduction.n_components,
                variance_threshold=self.config.dimensionality_reduction.variance_threshold,
                apply_after_generation=self.config.dimensionality_reduction.apply_after_generation,
            ),

            models=ResolvedModelSpaceConfig(
                include_models=self.config.models.include_models,
                exclude_models=self.config.models.exclude_models,
                top_k_models=self.config.models.top_k_models,
                compute_budget=self.config.search.compute_budget,
            ),

            search=ResolvedSearchConfig(
                scout_sample_fraction=self.config.search.scout_sample_fraction,
                scout_folds=self.config.search.scout_folds,
                time_budget_soft=self.config.search.time_budget_soft,
                min_improvement_over_dummy=self.config.search.min_improvement_over_dummy,
            ),

            data_quality=ResolvedDataQualityConfig(
                leak_handling=self.config.data_quality.leak_handling,
                id_threshold=self.config.data_quality.id_threshold,
            ),

            runtime=ResolvedRuntimeConfig(
                seed=self.config.runtime.seed,
                n_jobs=self.config.runtime.n_jobs,
                log=self.config.runtime.log,
            ),

            optuna=ResolvedOptunaConfig(
                enabled=self.config.optuna.enabled,
                n_trials=self.config.optuna.n_trials,
                direction=self.config.optuna.direction,
                n_jobs=self.config.optuna.n_jobs,
                seed=self.config.optuna.seed,
            ),

            artifacts=ResolvedArtifactsConfig(
                models=models,
                cv_object=cast(BaseCrossValidator, None),
                data_info=data_info,
                label_encoder=label_encoder,
                leaks=leaks,
            ),

            sampling=sampling_config,

            generate_optuna_plots=self.config.generate_optuna_plots,
            display_optuna_plots=self.config.display_optuna_plots,
        )

        # ───────── Phase 2: Derive CV object ─────────
        resolved.artifacts.cv_object = get_cv_object(y, resolved)

        return X, y, resolved

    @staticmethod
    def _filter_models(
        models: dict[str, ModelSpec],
        data_info: DataInfo,
        config: AutoMLConfig,
    ) -> dict[str, ModelSpec]:

        ms_config: ModelSpaceConfig = config.models
        s_config: SearchConfig = config.search

        if ms_config.include_models:
            models = {
                name: info
                for name, info in models.items()
                if name in ms_config.include_models
            }

        if ms_config.exclude_models:
            models = {
                name: info
                for name, info in models.items()
                if name not in ms_config.exclude_models
            }

        models = {
            name: info
            for name, info in models.items()
            if is_model_suitable(info, data_info)
        }

        if s_config.compute_budget == "low":
            models = {
                name: info
                for name, info in models.items()
                if info.training_cost == COST_LOW
            }

        elif s_config.compute_budget == "medium":
            models = {
                name: info
                for name, info in models.items()
                if info.training_cost in (COST_LOW, COST_MEDIUM)
            }

        if not models:
            raise ValueError("No models available after filtering.")

        return models
