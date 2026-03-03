# optimization/search_executor.py

from sklearn.model_selection import GridSearchCV


def run_grid_search(pipeline, param_grid, config, X, y, cv):

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=config.metric,
        n_jobs=config.n_jobs,
        refit=True
    )

    search.fit(X, y)

    return (
        search.best_estimator_,
        search.best_score_,
        search.best_params_
    )
