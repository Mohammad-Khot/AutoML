# core/state.py
import math


class AutoMLState:
    def __init__(self):
        self.scores: dict[str, float] = {}

    def update(self, model_name: str, score: float) -> None:
        if not isinstance(score, (int, float)):
            raise TypeError("score must be numeric")
        if math.isnan(score):
            raise ValueError("score cannot be NaN")
        self.scores[model_name] = float(score)

    def best(self) -> str | None:
        return max(self.scores, key=self.scores.get) if self.scores else None

    def as_sorted(self) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
