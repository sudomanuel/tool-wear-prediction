"""
layered_pipeline.py — engine del pipeline experimental por capas (LOEO-only).

Arbol:
    D (dataset)
    ├── N (real)
    │   ├── ST           defaults
    │   ├── CT_random    RandomizedSearchCV(n_iter=20)
    │   └── CT_grid      GridSearchCV (grids moderados)
    └── A (augmented por strategy)
        ├── ST
        ├── CT_random
        └── CT_grid

Cada hoja se evalua exclusivamente con **LOEO-CV** (10 folds, n_test=1).
Hold-out fue eliminado del flujo principal porque con n_test=2 sus
metricas son altamente inestables (R² puede oscilar de 0.9 a -2 con otro
seed). LOEO-CV es la metrica honesta del proyecto.

Tuning:
  - El tuning se hace una sola vez por (modelo, rama) usando GroupKFold
    sobre TODOS los datos disponibles (los 10 experimentos).
  - Los mejores hiperparametros se refittean en cada fold LOEO (compromiso
    documentado en methodology_notes.md: no es nested-CV completo).
  - El tuning "ve" los mismos experimentos que LOEO evalua, generando un
    sesgo optimista menor que el coste de nested-CV completo con n=10.
"""
from __future__ import annotations

import warnings
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, GridSearchCV

from .config import (
    EXPERIMENT_ID_COL, TARGET_COLUMN, RANDOM_SEED,
    N_AUGMENTED_PER_EXPERIMENT, AUGMENTATION_NOISE_SIGMA, AUGMENTATION_SCALING_RANGE,
)
from .modeling import (
    build_dummy, build_ridge, build_lasso, build_elasticnet,
    build_svr, build_rf, build_xgb, build_mlp, XGBOOST_AVAILABLE,
)
from .augmentation import augment_train
from .evaluation import compute_metrics, make_predictions_df
from .splitting import loeo_iter
from .tuning import get_param_distributions


# =============================================================================
# Constantes
# =============================================================================
AUGMENTATION_STRATEGIES = ['feature_noise', 'feature_scaling', 'grouped_scaling']
TUNABLE_MODELS = ['Ridge', 'Lasso', 'ElasticNet', 'SVR', 'RandomForest', 'XGBoost']
NONLINEAR_NAMES = {'svr', 'randomforest', 'xgboost', 'mlp'}


def all_baseline_builders() -> Dict[str, callable]:
    """Devuelve dict {nombre: builder()} — un constructor por modelo."""
    out = {
        'DummyRegressor': build_dummy,
        'Ridge':          build_ridge,
        'Lasso':          build_lasso,
        'ElasticNet':     build_elasticnet,
        'SVR':            build_svr,
        'RandomForest':   build_rf,
    }
    if XGBOOST_AVAILABLE:
        out['XGBoost'] = build_xgb
    out['MLP'] = build_mlp
    return out


# =============================================================================
# Grids reducidos para GridSearchCV
# =============================================================================
def get_grid_params(model_name: str) -> dict:
    """Grids para GridSearchCV (cartesianos completos, XGBoost reducido)."""
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
            'model__gamma':   ['scale', 'auto'],
        }
    if n == 'randomforest':
        return {
            'model__n_estimators':     [50, 100, 200],
            'model__max_depth':        [2, 3, 5, None],
            'model__min_samples_leaf': [1, 2, 3],
        }
    if n == 'xgboost':
        return {
            'model__n_estimators':  [50, 100, 200],
            'model__max_depth':     [1, 2, 3],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample':     [0.8],
            'model__colsample_bytree': [0.8],
            'model__reg_lambda':    [5, 10],
        }
    return {}


# =============================================================================
# Tuning helpers (sin HO: usa GroupKFold sobre TODOS los datos)
# =============================================================================
def _tune_one(name: str, pipe, X_all, y_all, groups_all,
              method: str, n_iter: int = 20, cv_splits: int = 5
              ) -> Tuple[dict, float, dict]:
    """
    Tune `pipe` con metodo ∈ {'random', 'grid'} usando GroupKFold sobre
    TODOS los datos (n=10). Devuelve (best_params, best_cv_mae, info).
    Si el modelo no es tuneable, devuelve dict vacio.

    NOTA metodologica: la busqueda ve los mismos experimentos que LOEO
    evaluara. Esto produce un sesgo optimista menor pero documentado.
    El alternative — nested-CV con re-tuning en cada fold — multiplica
    compute por 10 y con n=10 amplifica sobreajuste sin aportar.
    """
    if method == 'none' or name not in TUNABLE_MODELS:
        return {}, float('nan'), {'method': 'none'}

    if method == 'random':
        space = get_param_distributions(name)
    elif method == 'grid':
        space = get_grid_params(name)
    else:
        raise ValueError(f"method desconocido: {method}")

    if not space:
        return {}, float('nan'), {'method': method, 'note': 'empty_space'}

    n_groups = len(np.unique(groups_all))
    cv = GroupKFold(n_splits=min(cv_splits, n_groups))
    splitter = list(cv.split(X_all, y_all, groups=groups_all))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if method == 'random':
            search = RandomizedSearchCV(
                estimator=clone(pipe),
                param_distributions=space,
                n_iter=n_iter,
                cv=splitter,
                scoring='neg_mean_absolute_error',
                random_state=RANDOM_SEED,
                n_jobs=-1,
                refit=False,
            )
        else:
            search = GridSearchCV(
                estimator=clone(pipe),
                param_grid=space,
                cv=splitter,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                refit=False,
            )
        search.fit(X_all, y_all)

    return (search.best_params_,
            -float(search.best_score_),
            {'method': method, 'n_candidates': len(search.cv_results_['params'])})


def _apply_params(pipe, best_params: dict):
    """Devuelve clone(pipe).set_params(**best_params)."""
    new = clone(pipe)
    if best_params:
        new.set_params(**best_params)
    return new


# =============================================================================
# Augmentation helper
# =============================================================================
def _augment_or_keep(train_df: pd.DataFrame, aug_strategy: str,
                     seed: int) -> Tuple[pd.DataFrame, int, int]:
    """Devuelve (df_resultante, n_original, n_aug). Si aug_strategy=='none', no toca."""
    if aug_strategy == 'none':
        return train_df, len(train_df), 0
    aug_df = augment_train(
        train_df, strategy=aug_strategy,
        n_augmented=N_AUGMENTED_PER_EXPERIMENT,
        noise_sigma=AUGMENTATION_NOISE_SIGMA,
        scaling_range=AUGMENTATION_SCALING_RANGE,
        seed=seed,
    )
    n_orig = int((~aug_df.get('is_augmented', pd.Series([False] * len(aug_df)))).sum()) \
             if 'is_augmented' in aug_df.columns else len(train_df)
    n_aug = int(aug_df['is_augmented'].sum()) if 'is_augmented' in aug_df.columns else 0
    return aug_df, n_orig, n_aug


# =============================================================================
# Ejecucion de una rama (solo LOEO)
# =============================================================================
def run_branch(branch_id: str,
               data_branch: str,
               tuning_method: str,
               aug_strategy: str,
               full_df: pd.DataFrame,
               feat_cols: List[str],
               models_filter: Optional[List[str]] = None
               ) -> Dict:
    """
    Ejecuta una rama:
      1. Tuning (si CT) via GroupKFold sobre TODOS los datos.
      2. LOEO-CV con esos params refitteados en cada fold.

    Devuelve dict con metrics_rows, predictions_rows, tuning_rows y
    best_estimators_full (cada modelo entrenado en el dataset completo,
    util para SHAP).
    """
    print(f"\n>>> branch {branch_id}  (data={data_branch}, tuning={tuning_method}, aug={aug_strategy})")
    t0 = time.time()

    seed = RANDOM_SEED

    # Para tuning: usamos TODOS los datos (n=10), augmentados si A.
    # Para SHAP: necesitamos el estimador entrenado en datos reales (sin aug).
    tune_df = full_df
    if data_branch == 'A':
        tune_df, _, _ = _augment_or_keep(full_df, aug_strategy, seed)
    X_tune = tune_df[feat_cols].values.astype(float)
    y_tune = tune_df[TARGET_COLUMN].values.astype(float)
    groups_tune = tune_df[EXPERIMENT_ID_COL].astype(int).values

    builders = all_baseline_builders()
    if models_filter is not None:
        builders = {k: v for k, v in builders.items() if k in models_filter}

    metrics_rows = []
    predictions_rows = []
    tuning_rows = []
    best_estimators_full: Dict[str, object] = {}
    best_params_per_model: Dict[str, dict] = {}

    # ---------- Step 1: tuning de cada modelo (1 sola vez por rama) ----------
    for name, builder in builders.items():
        base_pipe = builder()
        if base_pipe is None:
            continue
        try:
            best_params, cv_mae, tune_info = _tune_one(
                name, base_pipe, X_tune, y_tune, groups_tune, tuning_method
            )
        except Exception as exc:
            warnings.warn(f"[{branch_id}/TUNE/{name}] {exc}")
            best_params, cv_mae, tune_info = {}, float('nan'), {
                'method': tuning_method, 'error': str(exc)
            }
        best_params_per_model[name] = best_params
        tuning_rows.append({
            'model': name,
            'data_branch': data_branch,
            'tuning_method': tuning_method if name in TUNABLE_MODELS else 'none',
            'augmentation_strategy': aug_strategy,
            'branch_id': branch_id,
            'best_params': json.dumps({k: _safe(v) for k, v in best_params.items()}),
            'best_cv_score_mae': cv_mae,
            'scoring': 'neg_mean_absolute_error',
            'cv_strategy': f'GroupKFold(k=5) sobre {len(np.unique(groups_tune))} experimentos',
            'notes': json.dumps(tune_info),
        })

    # ---------- Step 2: LOEO-CV ----------
    loeo_y_true = {name: [] for name in builders}
    loeo_y_pred = {name: [] for name in builders}
    loeo_eids   = {name: [] for name in builders}
    loeo_folds  = {name: [] for name in builders}
    loeo_n_train = {name: [] for name in builders}

    fold_idx = 0
    for tr_df_fold, te_df_fold in loeo_iter(full_df, group_col=EXPERIMENT_ID_COL):
        fold_idx += 1
        tr_aug, n_o, n_a = _augment_or_keep(tr_df_fold, aug_strategy, seed + fold_idx)
        X_tr = tr_aug[feat_cols].values.astype(float)
        y_tr = tr_aug[TARGET_COLUMN].values.astype(float)
        X_te = te_df_fold[feat_cols].values.astype(float)
        y_te = te_df_fold[TARGET_COLUMN].values.astype(float)
        eid_te = te_df_fold[EXPERIMENT_ID_COL].astype(int).values

        for name, builder in builders.items():
            base_pipe = builder()
            if base_pipe is None:
                continue
            try:
                if tuning_method == 'none' or name not in TUNABLE_MODELS:
                    est = clone(base_pipe)
                else:
                    est = _apply_params(base_pipe, best_params_per_model.get(name, {}))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    est.fit(X_tr, y_tr)
                    y_p = est.predict(X_te)
            except Exception as exc:
                warnings.warn(f"[{branch_id}/LOEO/fold{fold_idx}/{name}] {exc}")
                y_p = np.full_like(y_te, np.nan, dtype=float)

            loeo_y_true[name].extend(y_te.tolist())
            loeo_y_pred[name].extend(y_p.tolist())
            loeo_eids[name].extend(eid_te.tolist())
            loeo_folds[name].extend([fold_idx] * len(y_te))
            loeo_n_train[name].append(int(len(y_tr)))

    # ---------- Step 3: agregacion + entrenamiento final para SHAP ----------
    for name, builder in builders.items():
        yt = np.array(loeo_y_true[name], dtype=float)
        yp = np.array(loeo_y_pred[name], dtype=float)
        if np.isfinite(yp).any():
            mets = compute_metrics(yt[np.isfinite(yp)], yp[np.isfinite(yp)])
        else:
            mets = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan}
        n_train_typ = int(np.median(loeo_n_train[name])) if loeo_n_train[name] else 0
        metrics_rows.append({
            'model': name,
            'data_branch': data_branch,
            'tuning_branch': 'CT' if tuning_method != 'none' else 'ST',
            'tuning_method': tuning_method if name in TUNABLE_MODELS else 'none',
            'validation_type': 'loeo',
            'augmentation_strategy': aug_strategy,
            'branch_id': branch_id,
            **mets,
            'n_train': n_train_typ,
            'n_train_original': (len(full_df) - 1),
            'n_train_augmented': max(0, n_train_typ - (len(full_df) - 1)),
            'n_test': len(yt),
            'n_folds': fold_idx,
            'notes': '',
        })
        pdf = make_predictions_df(name, loeo_eids[name], yt, yp,
                                  extra={
                                      'data_branch': data_branch,
                                      'tuning_method': tuning_method if name in TUNABLE_MODELS else 'none',
                                      'validation_type': 'loeo',
                                      'augmentation_strategy': aug_strategy,
                                      'branch_id': branch_id,
                                  })
        pdf.insert(2, 'fold_id', loeo_folds[name])
        predictions_rows.append(pdf)

        # Entrenar estimador final sobre dataset completo (datos REALES) para SHAP.
        # Para A: SHAP igualmente se calcula sobre datos REALES, entrenamos
        # con el mismo augmentation que recibio la rama (consistente con
        # tuning), pero filtramos augmented al explicar (lo hace shap_analysis).
        base_pipe = builder()
        if base_pipe is None:
            continue
        try:
            if tuning_method == 'none' or name not in TUNABLE_MODELS:
                est_full = clone(base_pipe)
            else:
                est_full = _apply_params(base_pipe, best_params_per_model.get(name, {}))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                est_full.fit(X_tune, y_tune)
            best_estimators_full[name] = est_full
        except Exception as exc:
            warnings.warn(f"[{branch_id}/FULL_FIT/{name}] {exc}")

    print(f"<<< branch {branch_id} OK  ({time.time() - t0:.1f}s, "
          f"n_models={len(builders)})")

    return {
        'metrics_rows': metrics_rows,
        'predictions_rows': predictions_rows,
        'tuning_rows': tuning_rows,
        'best_estimators_full': best_estimators_full,
        'best_params_per_model': best_params_per_model,
    }


# =============================================================================
# Plan de ramas
# =============================================================================
def enumerate_branches() -> List[dict]:
    """Devuelve las 12 ramas: N x {ST, CT_random, CT_grid} +
       A x {ST, CT_random, CT_grid} x {3 estrategias}."""
    out = []
    for tm in ('none', 'random', 'grid'):
        suffix = 'ST' if tm == 'none' else f'CT_{tm}'
        out.append({
            'branch_id': f'N_{suffix}',
            'data_branch': 'N',
            'tuning_method': tm,
            'aug_strategy': 'none',
        })
    for aug in AUGMENTATION_STRATEGIES:
        for tm in ('none', 'random', 'grid'):
            suffix = 'ST' if tm == 'none' else f'CT_{tm}'
            out.append({
                'branch_id': f'A_{suffix}_{aug}',
                'data_branch': 'A',
                'tuning_method': tm,
                'aug_strategy': aug,
            })
    return out


# =============================================================================
# Final ranking (LOEO only)
# =============================================================================
def build_final_ranking(all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Ranking final ordenado por MAE LOEO (todas las filas son LOEO)."""
    df = all_metrics_df.copy()
    df = df[df['validation_type'] == 'loeo']
    df = df.sort_values(['MAE', 'RMSE', 'R2'],
                        ascending=[True, True, False]).reset_index(drop=True)
    df.insert(0, 'rank', np.arange(1, len(df) + 1))

    def _note(r):
        if not np.isfinite(r['MAE']):
            return 'failed'
        notes = ['LOEO-CV (honest, n=10)']
        if r['augmentation_strategy'] != 'none':
            notes.append(f"aug:{r['augmentation_strategy']}")
        if r['tuning_method'] != 'none':
            notes.append(f"tuning:{r['tuning_method']}")
        return '; '.join(notes)

    df['interpretation_note'] = df.apply(_note, axis=1)
    cols = ['rank', 'model', 'data_branch', 'tuning_method', 'validation_type',
            'augmentation_strategy', 'MAE', 'RMSE', 'R2', 'MAPE_%',
            'interpretation_note', 'branch_id']
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


# =============================================================================
# Resumenes derivados (delta, tuning effect, augmentation effect, random vs grid)
# =============================================================================
def build_branch_best_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mejor (modelo, metrica) por rama, ordenado por MAE."""
    df = metrics_df[metrics_df['validation_type'] == 'loeo'].dropna(subset=['MAE']).copy()
    if df.empty:
        return pd.DataFrame()
    best = df.sort_values('MAE').groupby('branch_id', as_index=False).first()
    cols = ['branch_id', 'data_branch', 'tuning_method', 'augmentation_strategy',
            'model', 'MAE', 'RMSE', 'R2', 'MAPE_%']
    return best[[c for c in cols if c in best.columns]].sort_values('MAE').reset_index(drop=True)


def build_delta_vs_baseline(metrics_df: pd.DataFrame,
                            baseline_branch: str = 'N_ST') -> pd.DataFrame:
    """delta_MAE = best_MAE_branch - best_MAE_baseline (negativo = mejora)."""
    best = build_branch_best_summary(metrics_df)
    if best.empty or baseline_branch not in best['branch_id'].values:
        return pd.DataFrame()
    base = best[best['branch_id'] == baseline_branch].iloc[0]
    out = best.copy()
    for m in ('MAE', 'RMSE', 'R2', 'MAPE_%'):
        if m in out.columns:
            out[f'delta_{m}_vs_baseline'] = out[m] - base[m]
    out['baseline_branch'] = baseline_branch

    def _interp(d):
        if abs(d) < 1.0:
            return 'practical tie (<1 µm)'
        if abs(d) < 5.0:
            return 'marginal (<5 µm, no significativo con n=10)'
        return 'mejora' if d < 0 else 'empeora'

    out['mae_interpretation'] = out['delta_MAE_vs_baseline'].apply(_interp)
    return out


def build_tuning_effect_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Por (data_branch, aug_strategy): comparar ST vs CT_random vs CT_grid."""
    df = metrics_df[metrics_df['validation_type'] == 'loeo'].dropna(subset=['MAE']).copy()
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(['data_branch', 'augmentation_strategy', 'tuning_method'])
    out = g['MAE'].min().reset_index().rename(columns={'MAE': 'best_MAE'})
    pivot = out.pivot(index=['data_branch', 'augmentation_strategy'],
                       columns='tuning_method', values='best_MAE').reset_index()
    for col in ('none', 'random', 'grid'):
        if col not in pivot.columns:
            pivot[col] = float('nan')
    pivot['delta_random_vs_none'] = pivot['random'] - pivot['none']
    pivot['delta_grid_vs_none']   = pivot['grid']   - pivot['none']
    pivot['delta_grid_vs_random'] = pivot['grid']   - pivot['random']
    pivot = pivot.rename(columns={
        'none': 'best_MAE_ST',
        'random': 'best_MAE_CT_random',
        'grid': 'best_MAE_CT_grid',
    })
    return pivot.reset_index(drop=True)


def build_augmentation_effect_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Por (tuning_method): comparar normal vs cada estrategia de augmentation."""
    df = metrics_df[metrics_df['validation_type'] == 'loeo'].dropna(subset=['MAE']).copy()
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(['tuning_method', 'data_branch', 'augmentation_strategy'])
    out = g['MAE'].min().reset_index().rename(columns={'MAE': 'best_MAE'})

    rows = []
    for tm in out['tuning_method'].unique():
        sub = out[out['tuning_method'] == tm]
        n_mae = sub[sub['data_branch'] == 'N']['best_MAE']
        baseline = float(n_mae.iloc[0]) if not n_mae.empty else float('nan')
        for _, r in sub.iterrows():
            rows.append({
                'tuning_method': tm,
                'data_branch':   r['data_branch'],
                'augmentation_strategy': r['augmentation_strategy'],
                'best_MAE': r['best_MAE'],
                'best_MAE_baseline_N': baseline,
                'delta_vs_N': r['best_MAE'] - baseline,
            })
    return pd.DataFrame(rows)


def build_random_vs_grid_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Comparar RandomizedSearchCV vs GridSearchCV en cada (data, aug)."""
    df = metrics_df[metrics_df['validation_type'] == 'loeo'].dropna(subset=['MAE']).copy()
    df = df[df['tuning_method'].isin(['random', 'grid'])].copy()
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(['data_branch', 'augmentation_strategy', 'tuning_method'])['MAE'].min()
    out = g.reset_index().pivot(
        index=['data_branch', 'augmentation_strategy'],
        columns='tuning_method', values='MAE'
    ).reset_index()
    for col in ('random', 'grid'):
        if col not in out.columns:
            out[col] = float('nan')
    out = out.rename(columns={'random': 'best_MAE_random', 'grid': 'best_MAE_grid'})
    out['delta_grid_minus_random'] = out['best_MAE_grid'] - out['best_MAE_random']
    out['winner'] = out.apply(
        lambda r: 'grid' if r['delta_grid_minus_random'] < -1.0
        else ('random' if r['delta_grid_minus_random'] > 1.0
              else 'practical_tie (<1 µm)'),
        axis=1
    )
    return out


# =============================================================================
# Evolucion del modelo: 12 ramas en orden, rama por rama
# =============================================================================
# Orden visual de las 12 ramas. Agrupa por "familia" para que se lea
# como una progresion: primero N (sin/con tuning), luego A_ST por
# estrategia, luego A_CT_random por estrategia, luego A_CT_grid.
BRANCH_PROGRESSION = [
    # (stage_order, branch_id, family, short_label)
    (1,  'N_ST',                          'N',           '1. N_ST'),
    (2,  'N_CT_random',                   'N · tuning',  '2. N_CT_random'),
    (3,  'N_CT_grid',                     'N · tuning',  '3. N_CT_grid'),
    (4,  'A_ST_feature_noise',            'A · ST',      '4. A_ST_fnoise'),
    (5,  'A_ST_feature_scaling',          'A · ST',      '5. A_ST_fscale'),
    (6,  'A_ST_grouped_scaling',          'A · ST',      '6. A_ST_gscale'),
    (7,  'A_CT_random_feature_noise',     'A · random',  '7. A_CTr_fnoise'),
    (8,  'A_CT_random_feature_scaling',   'A · random',  '8. A_CTr_fscale'),
    (9,  'A_CT_random_grouped_scaling',   'A · random',  '9. A_CTr_gscale'),
    (10, 'A_CT_grid_feature_noise',       'A · grid',    '10. A_CTg_fnoise'),
    (11, 'A_CT_grid_feature_scaling',     'A · grid',    '11. A_CTg_fscale'),
    (12, 'A_CT_grid_grouped_scaling',     'A · grid',    '12. A_CTg_gscale'),
]

FAMILY_COLORS = {
    'N':          '#1F4E79',
    'N · tuning': '#2E86AB',
    'A · ST':     '#D7906A',
    'A · random': '#A0521E',
    'A · grid':   '#7B2D26',
}


def _interpret_delta(d: float, eps_tie: float = 1.0, eps_marg: float = 5.0,
                     lower_is_better: bool = True) -> str:
    """Convierte un delta numerico en una etiqueta legible."""
    if not np.isfinite(d):
        return 'n/a'
    sign = 1 if lower_is_better else -1
    s = d * sign
    if abs(d) < eps_tie:
        return 'negligible change (<1 µm)'
    if abs(d) < eps_marg:
        return ('marginal improvement (<5 µm, not significant with n=10)' if s < 0
                else 'marginal worsening (<5 µm, not significant with n=10)')
    return 'improved' if s < 0 else 'worsened'


def build_model_evolution_summary(metrics_df: pd.DataFrame
                                  ) -> pd.DataFrame:
    """
    Para cada una de las **12 ramas** en orden de progresion del pipeline,
    elige el modelo con mejor MAE LOEO y reporta metricas + delta vs
    rama previa y vs baseline (N_ST). 12 filas, una por rama.

    No asume mejora monotonica. La interpretacion etiqueta cada Δ como
    `negligible change (<1 µm)`, `marginal worsening`, `improved`, etc.
    """
    df = metrics_df[metrics_df['validation_type'] == 'loeo'].dropna(subset=['MAE']).copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    baseline_mae = float('nan')
    prev_mae = float('nan')

    for stage_order, bid, family, label in BRANCH_PROGRESSION:
        sub = df[df['branch_id'] == bid]
        if sub.empty:
            rows.append({
                'stage_order': stage_order, 'branch_id': bid,
                'family': family, 'short_label': label,
                'model': '',
                'MAE': float('nan'), 'RMSE': float('nan'),
                'R2': float('nan'), 'MAPE_%': float('nan'),
                'delta_MAE_vs_previous_stage': float('nan'),
                'delta_MAE_vs_baseline': float('nan'),
                'interpretation': 'no data',
            })
            continue
        best = sub.sort_values('MAE').iloc[0]

        if stage_order == 1:
            baseline_mae = float(best['MAE'])
            interp = 'baseline'
            delta_prev = 0.0
            delta_base = 0.0
        else:
            delta_prev = float(best['MAE']) - prev_mae
            delta_base = float(best['MAE']) - baseline_mae
            d_label = _interpret_delta(delta_prev, lower_is_better=True)
            interp = f"vs previous: {d_label}"

        rows.append({
            'stage_order': stage_order, 'branch_id': bid,
            'family': family, 'short_label': label,
            'model': best['model'],
            'MAE':    float(best['MAE']),
            'RMSE':   float(best['RMSE']),
            'R2':     float(best['R2']),
            'MAPE_%': float(best['MAPE_%']),
            'delta_MAE_vs_previous_stage': float(delta_prev),
            'delta_MAE_vs_baseline':       float(delta_base),
            'interpretation': interp,
        })
        prev_mae = float(best['MAE'])

    out = pd.DataFrame(rows)
    if not out.empty and out['MAE'].notna().any():
        idx_best = out['MAE'].idxmin()
        out.loc[idx_best, 'interpretation'] = (
            out.loc[idx_best, 'interpretation'] + '  ·  BEST OVERALL'
        )
    return out


def build_model_evolution_by_model(metrics_df: pd.DataFrame,
                                   top_n_models: int = 3
                                   ) -> pd.DataFrame:
    """
    Una fila por (modelo, rama). Cada modelo recorre las 12 ramas en orden.
    Devuelve top_n_models (segun MAE en N_ST baseline).
    """
    df = metrics_df[metrics_df['validation_type'] == 'loeo'].dropna(subset=['MAE']).copy()
    if df.empty:
        return pd.DataFrame()

    baseline = df[df['branch_id'] == 'N_ST'].sort_values('MAE')
    if baseline.empty:
        return pd.DataFrame()
    top_models = baseline['model'].head(top_n_models).tolist()

    rows = []
    for model in top_models:
        for stage_order, bid, family, label in BRANCH_PROGRESSION:
            sub = df[(df['model'] == model) & (df['branch_id'] == bid)]
            if sub.empty:
                rows.append({
                    'model': model, 'stage_order': stage_order,
                    'branch_id': bid, 'family': family, 'short_label': label,
                    'MAE': float('nan'), 'RMSE': float('nan'),
                    'R2': float('nan'), 'MAPE_%': float('nan'),
                })
                continue
            best = sub.sort_values('MAE').iloc[0]
            rows.append({
                'model': model, 'stage_order': stage_order,
                'branch_id': bid, 'family': family, 'short_label': label,
                'MAE':    float(best['MAE']),
                'RMSE':   float(best['RMSE']),
                'R2':     float(best['R2']),
                'MAPE_%': float(best['MAPE_%']),
            })
    return pd.DataFrame(rows)


def select_predictions_for_multi_overlay(predictions_df: pd.DataFrame,
                                         rank_df: pd.DataFrame
                                         ) -> list:
    """
    Selecciona 5 configuraciones representativas para el scatter overlay:
      1. baseline           = mejor modelo en N_ST
      2. mejor tuneado N    = mejor (modelo, rama) entre N_CT_*
      3. mejor A_ST         = mejor entre A_ST_*
      4. mejor A + tuneado  = mejor entre A_CT_*
      5. best global        = top 1 del ranking
    Devuelve lista de dicts con {label, model, branch_id, color, y_real, y_pred, eids}.
    """
    if predictions_df.empty or rank_df.empty:
        return []
    df = predictions_df[predictions_df['validation_type'] == 'loeo'].copy()

    def _pick(branch_filter, label, color):
        sub_r = rank_df.dropna(subset=['MAE'])
        if isinstance(branch_filter, str):
            sub_r = sub_r[sub_r['branch_id'] == branch_filter]
        else:
            sub_r = sub_r[sub_r['branch_id'].apply(branch_filter)]
        if sub_r.empty:
            return None
        row = sub_r.iloc[0]
        sub_p = df[(df['model'] == row['model']) & (df['branch_id'] == row['branch_id'])]
        if sub_p.empty:
            return None
        return {
            'label': label,
            'model': row['model'],
            'branch_id': row['branch_id'],
            'color': color,
            'y_real': sub_p['VB_real'].values,
            'y_pred': sub_p['VB_pred'].values,
            'eids':   sub_p['experiment_id'].values,
            'mae':    float(row['MAE']),
        }

    selections = []
    s1 = _pick('N_ST', '1. Baseline (N_ST)', '#1F4E79')
    s2 = _pick(lambda b: b in ('N_CT_random', 'N_CT_grid'),
               '2. Mejor tuneado (N_CT)', '#2E86AB')
    s3 = _pick(lambda b: b.startswith('A_ST_'),
               '3. Mejor A_ST', '#D7906A')
    s4 = _pick(lambda b: b.startswith('A_CT_'),
               '4. Mejor A_CT (aug+tuning)', '#A0521E')
    s5 = _pick(lambda b: True,
               '5. BEST GLOBAL', '#D7263D')
    for s in (s1, s2, s3, s4, s5):
        if s is not None:
            # Deduplicar si "best global" coincide con otra ya seleccionada
            tag = (s['model'], s['branch_id'])
            if any((x['model'], x['branch_id']) == tag and 'BEST' not in x['label']
                   for x in selections) and 'BEST' in s['label']:
                # En vez de duplicar, anotamos el ganador en una etiqueta extra
                for x in selections:
                    if (x['model'], x['branch_id']) == tag:
                        x['label'] = x['label'] + '  =  BEST GLOBAL'
                        x['color'] = '#D7263D'
                continue
            selections.append(s)
    return selections


# =============================================================================
# Utilidades
# =============================================================================
def _safe(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if v is None:
        return None
    return str(v)
