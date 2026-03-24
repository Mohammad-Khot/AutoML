import numpy as np


def compute_bootstrap_ci(
        scores,
        confidence=0.95,
        n_bootstrap=1000,
        seed=None,
):
    scores = np.array(scores)
    n = len(scores)

    if n == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)

    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)

    mean = float(np.mean(scores))
    margin = float((upper - lower) / 2)

    return mean, margin
