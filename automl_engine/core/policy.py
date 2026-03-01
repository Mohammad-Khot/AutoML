TOL = 1e-6


def select_best_model(scores: dict, priority: dict[str, int]) -> str:
    if not scores:
        raise ValueError("No model could be evaluated successfully.")

    # Detect score format
    first_value = next(iter(scores.values()))

    if isinstance(first_value, dict):
        # dict-of-dict format
        max_score = max(v["mean_score"] for v in scores.values())
        tied = [
            name for name, v in scores.items()
            if abs(v["mean_score"] - max_score) <= TOL
        ]
    else:
        # flat dict format
        max_score = max(scores.values())
        tied = [
            name for name, score in scores.items()
            if abs(score - max_score) <= TOL
        ]

    return min(tied, key=lambda name: (priority.get(name, 99), name))
