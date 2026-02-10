

def select_best_model(scores: dict, priority: dict):
    """
    Select model based on:
    1. highest score
    2. lowest priority value
    3. stable name sort
    """
    if not scores:
        raise ValueError(
            "No model could be evaluated successfully."
            "Dataset too small or CV impossible"
        )
    max_score = max(scores.values())

    tied = [
        name for name, score in scores.items()
        if score == max_score
    ]

    if len(tied) == 1:
        return tied[0]

    tied.sort(key=lambda name: (priority.get(name, 99), name))
    return tied[0]
