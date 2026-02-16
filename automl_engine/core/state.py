# core/state.py
import math


class AutoMLState:
    def __init__(self):
        self.models: dict[str, dict] = {}

    def update(
        self,
        model_name: str,
        score: float,
        pipeline=None,
        params=None
    ) -> None:

        if not isinstance(score, (int, float)):
            raise TypeError("score must be numeric")

        if math.isnan(score):
            raise ValueError("score cannot be NaN")

        self.models[model_name] = {
            "score": float(score),
            "pipeline": pipeline,
            "params": params,
        }

    @property
    def scores(self) -> dict[str, float]:
        return {
            name: info["score"]
            for name, info in self.models.items()
        }

    def get_pipeline(self, model_name: str):
        return self.models.get(model_name, {}).get("pipeline")

    def get_params(self, model_name: str):
        return self.models.get(model_name, {}).get("params")

    def best(self) -> str | None:
        return max(self.scores, key=self.scores.get) if self.scores else None

    def as_sorted(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
