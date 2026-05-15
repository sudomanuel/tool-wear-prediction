"""
tuning.py — tuning ligero con RandomizedSearchCV.

Para XGBoost se ofrece adicionalmente un GridSearchCV pequeno alrededor de
los mejores parametros (refinamiento opcional). Si falla por cualquier
motivo, se omite con warning y se mantiene el resultado de Random.

Cross-validation: GroupKFold con groups=experiment_id (anti-leakage).
Scoring: neg_mean_absolute_error (priorizamos MAE).
"""
import warnings
import numpy as np
from typing import Tuple
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV

from .config import RANDOM_SEED


# -----------------------------------------------------------------------------
# Espacios de busqueda — moderados
# -----------------------------------------------------------------------------
def get_param_distributions(model_name: str) -> dict:
    """
    Devuelve el espacio de busqueda. Los nombres usan el prefijo del step
    'model__' por estar dentro de un Pipeline.
    """
    n = model_name.lower()
    if n == 'ridge':
        return {'model__alpha': [0.01, 0.1, 1, 10, 100]}
    if n == 'lasso':
        return {'model__alpha': [0.001, 0.01, 0.1, 1, 10]}
    if n == 'elasticnet':
        return {
            'model__alpha':    [0.01, 0.1, 1, 10],
            'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        }
    if n == 'svr':
        return {
            'model__C':       [0.1, 1, 10, 100],
            'model__epsilon': [0.1, 1, 5, 10],
            'model__gamma':   ['scale', 'auto', 0.01, 0.1],
        }
    if n == 'randomforest':
        return {
            'model__n_estimators':     [50, 100, 200],
            'model__max_depth':        [2, 3, 5, None],
            'model__min_samples_leaf': [1, 2, 3],
        }
    if n == 'xgboost':
        return {
            'model__n_estimators':     [30, 50, 100, 200],
            'model__max_depth':        [1, 2, 3],
            'model__learning_rate':    [0.01, 0.05, 0.1],
            'model__subsample':        [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.5, 0.8, 1.0],
            'model__reg_lambda':       [1, 5, 10, 20],
        }
    return {}


# -----------------------------------------------------------------------------
# Tuning principal
# -----------------------------------------------------------------------------
def tune_model(name: str, pipeline,
               X_train, y_train, groups_train,
               n_iter: int = 20,
               cv_splits: int = 5,
               scoring: str = 'neg_mean_absolute_error'):
    """
    Hace RandomizedSearchCV con GroupKFold sobre el train.
    Devuelve (best_estimator, best_params_dict, search_results_df).
    Si el espacio esta vacio o el modelo no soporta tuning, devuelve
    el pipeline original sin cambios.
    """
    space = get_param_distributions(name)
    if not space:
        return pipeline, {}, None

    n_groups = len(np.unique(groups_train))
    cv = GroupKFold(n_splits=min(cv_splits, n_groups))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            estimator=clone(pipeline),
            param_distributions=space,
            n_iter=n_iter,
            cv=cv.split(X_train, y_train, groups=groups_train),
            scoring=scoring,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
    import pandas as pd
    cv_res = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, search.best_params_, cv_res


def refine_xgb_grid(best_random_params: dict,
                    pipeline,
                    X_train, y_train, groups_train,
                    cv_splits: int = 5):
    """
    Construye una grilla pequena alrededor de los mejores params de XGBoost.
    Maximo 3x3 = 9 combinaciones por par (n_estimators, max_depth).
    """
    n_est = best_random_params.get('model__n_estimators', 100)
    md    = best_random_params.get('model__max_depth', 3)
    lr    = best_random_params.get('model__learning_rate', 0.05)

    grid = {
        'model__n_estimators':  sorted({max(20, int(n_est * 0.7)),
                                        int(n_est),
                                        int(n_est * 1.3)}),
        'model__max_depth':     sorted({max(1, md - 1), md, md + 1}),
        'model__learning_rate': sorted({max(0.005, lr / 2), lr, min(0.3, lr * 2)}),
    }
    # fijamos el resto en los mejores valores de random
    for k in ('model__subsample', 'model__colsample_bytree', 'model__reg_lambda'):
        if k in best_random_params:
            grid[k] = [best_random_params[k]]

    n_groups = len(np.unique(groups_train))
    cv = GroupKFold(n_splits=min(cv_splits, n_groups))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=grid,
            cv=cv.split(X_train, y_train, groups=groups_train),
            scoring='neg_mean_absolute_error',
            n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
    import pandas as pd
    return search.best_estimator_, search.best_params_, pd.DataFrame(search.cv_results_)
