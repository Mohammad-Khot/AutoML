TOL = 1e-6


def select_best_model(scores: dict[str, float], priority: dict[str, int]) -> str:
    if not scores:
        raise ValueError("No model could be evaluated successfully. Dataset too small or CV impossible")

    max_score = max(scores.values())

    tied = [n for n, s in scores.items() if abs(s - max_score) <= TOL]

    return min(tied, key=lambda n: (priority.get(n, 99), n))
