from sklearn.decomposition import PCA, FastICA, TruncatedSVD


def build_dimensionality_reduction(config):
    method = config.dimensionality_reduction.method

    if method in ("none", None):
        return "passthrough"

    n = config.dimensionality_reduction.n_components

    if method == "pca":
        return PCA(n_components=n)

    if method == "ica":
        return FastICA(n_components=n)

    if method == "svd":
        return TruncatedSVD(n_components=n)

    return "passthrough"
