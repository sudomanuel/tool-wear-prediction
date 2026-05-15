"""
shap_analysis.py — interpretabilidad SHAP encapsulada.

Reglas:
- SHAP NUNCA entrena modelos. Solo explica modelos ya entrenados.
- SHAP se calcula sobre filas REALES (sin augmentation).
- Selecciona el explainer correcto segun el tipo de modelo subyacente:
    * Linear (Ridge/Lasso/ElasticNet) -> LinearExplainer
    * Tree   (RandomForest/XGBoost)   -> TreeExplainer
    * Otro (SVR/MLP/Dummy)            -> fallback: coeficientes (lineales)
                                         o se omite con warning.
- Si algo falla, NO rompe el pipeline: warning + skip.

Salidas:
- outputs/metrics/shap/shap_feature_ranking_<model>.csv
- outputs/metrics/shap/shap_values_<model>.csv  (formato largo, una fila
  por (experiment_id, feature))
- outputs/figures/shap/shap_bar_<model>.png
- outputs/figures/shap/shap_summary_<model>.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("[SHAP] paquete 'shap' no instalado. SHAP se omitira.")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor

from .config import METRICS_SHAP, FIG_SHAP, FIGURE_DPI, FIGURE_FORMAT


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _final_estimator(pipe):
    """Devuelve (estimator_final, transformer_pre)."""
    if isinstance(pipe, Pipeline):
        # Aplicamos toda la cadena menos el ultimo paso para obtener X transformado.
        pre = Pipeline(pipe.steps[:-1]) if len(pipe.steps) > 1 else None
        est = pipe.steps[-1][1]
        return est, pre
    return pipe, None


def _is_linear(est) -> bool:
    return isinstance(est, (Ridge, Lasso, ElasticNet, LinearRegression))


def _is_tree(est) -> bool:
    name = type(est).__name__.lower()
    if isinstance(est, RandomForestRegressor):
        return True
    # XGBRegressor (no requiere import dependiente)
    return name in {'xgbregressor', 'lgbmregressor', 'extratreesregressor',
                    'gradientboostingregressor', 'randomforestregressor'}


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# API principal
# -----------------------------------------------------------------------------
def explain_model(model_name: str,
                  pipeline,
                  X_train_real: np.ndarray,
                  X_explain_real: np.ndarray,
                  feature_names: list,
                  explain_experiment_ids: list = None,
                  top_n: int = 20,
                  file_prefix: str = ''
                  ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calcula SHAP para `pipeline` ya entrenado y guarda CSVs + plots.
    Devuelve (ranking_df, values_long_df) o None si SHAP fallo.
    """
    if not SHAP_AVAILABLE:
        return None

    est, pre = _final_estimator(pipeline)

    # Transformamos los datos con el preprocesador del pipeline
    try:
        X_train_t   = pre.transform(X_train_real)   if pre is not None else X_train_real
        X_explain_t = pre.transform(X_explain_real) if pre is not None else X_explain_real
    except Exception as exc:
        warnings.warn(f"[SHAP] {model_name}: error al transformar X: {exc}")
        return None

    # Elegir explainer
    shap_values = None
    try:
        if _is_linear(est):
            explainer = shap.LinearExplainer(est, X_train_t, feature_names=feature_names)
            shap_values = explainer.shap_values(X_explain_t)
        elif _is_tree(est):
            explainer = shap.TreeExplainer(est)
            shap_values = explainer.shap_values(X_explain_t)
        else:
            # Fallback: KernelExplainer (muy lento) — lo intentamos con background pequeno
            bg = shap.sample(X_train_t, min(20, len(X_train_t)),
                              random_state=0)
            explainer = shap.KernelExplainer(est.predict, bg)
            shap_values = explainer.shap_values(X_explain_t, nsamples=100, silent=True)
    except Exception as exc:
        warnings.warn(f"[SHAP] {model_name}: explainer fallo: {exc}")
        return None

    if shap_values is None:
        return None

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    # --- ranking ---
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)
    ranking_df = pd.DataFrame({
        'model': model_name,
        'feature': feature_names,
        'mean_abs_shap': mean_abs,
        'mean_shap': mean_signed,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    ranking_df.insert(2, 'rank', np.arange(1, len(ranking_df) + 1))

    # --- valores largos ---
    explain_experiment_ids = list(explain_experiment_ids) if explain_experiment_ids is not None \
        else list(range(len(X_explain_real)))
    long_rows = []
    for i in range(shap_values.shape[0]):
        eid = explain_experiment_ids[i] if i < len(explain_experiment_ids) else i
        for j, fname in enumerate(feature_names):
            long_rows.append({
                'model': model_name,
                'experiment_id': int(eid) if isinstance(eid, (int, np.integer)) else eid,
                'feature': fname,
                'shap_value': float(shap_values[i, j]),
                'feature_value': float(X_explain_real[i, j])
                                  if not isinstance(X_explain_real[i, j], str) else None,
            })
    long_df = pd.DataFrame(long_rows)

    # --- guardar ---
    METRICS_SHAP.mkdir(parents=True, exist_ok=True)
    slug = _slug(model_name)
    pref = file_prefix
    ranking_path = METRICS_SHAP / f"{pref}shap_feature_ranking_{slug}.csv"
    values_path  = METRICS_SHAP / f"{pref}shap_values_{slug}.csv"
    ranking_df.to_csv(ranking_path, index=False)
    long_df.to_csv(values_path, index=False)

    # --- plots ---
    _plot_bar(ranking_df, model_name, top_n, file_prefix=pref)
    _plot_summary(shap_values, X_explain_t, feature_names,
                  ranking_df, model_name, top_n, file_prefix=pref)

    return ranking_df, long_df


def _slug(name: str) -> str:
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '')


def _filter_nonzero(ranking_df: pd.DataFrame, top_n: int,
                    eps: float = 1e-6) -> pd.DataFrame:
    """
    Filtra features con |SHAP| ≈ 0 (caso tipico de Lasso con muchos
    coeficientes anulados). Devuelve maximo top_n filas con importancia
    no-trivial. Si quedan menos de 3, devuelve top_n original como fallback.
    """
    nz = ranking_df[ranking_df['mean_abs_shap'] > eps]
    if len(nz) >= 3:
        return nz.head(top_n)
    return ranking_df.head(top_n)


def _plot_bar(ranking_df: pd.DataFrame, model_name: str, top_n: int,
              file_prefix: str = ''):
    top = _filter_nonzero(ranking_df, top_n).iloc[::-1]
    n_total = len(ranking_df)
    n_nz    = int((ranking_df['mean_abs_shap'] > 1e-6).sum())
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.32 * len(top))))
    bars = ax.barh(top['feature'], top['mean_abs_shap'],
                   color='#2E86AB', edgecolor='k', linewidth=0.4)
    # numero al final de cada barra
    xmax = top['mean_abs_shap'].max() if len(top) else 1.0
    for bar, v in zip(bars, top['mean_abs_shap']):
        ax.text(v + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3g}", va='center', ha='left', fontsize=8)
    ax.set_xlim(0, xmax * 1.15)
    ax.set_xlabel('|SHAP| medio')
    subtitle = (f"top {len(top)} de {n_total} features"
                + (f"  (no-cero: {n_nz})" if n_nz < n_total else ""))
    ax.set_title(f'SHAP — importancia de features ({model_name})\n{subtitle}',
                 fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, FIG_SHAP / f"{file_prefix}shap_bar_{_slug(model_name)}.{FIGURE_FORMAT}")


def _plot_summary(shap_values: np.ndarray, X_t, feature_names, ranking_df,
                  model_name: str, top_n: int, file_prefix: str = ''):
    """
    Summary (beeswarm) limitado a features con |SHAP| > 0. Si Lasso anulo
    muchos coeficientes, mostramos solo los que realmente afectan.
    """
    try:
        nz_mask = ranking_df['mean_abs_shap'].values > 1e-6
        # nombres no-cero ordenados por ranking original (rank=1 primero)
        nz_features = ranking_df.loc[nz_mask, 'feature'].head(top_n).tolist()
        if len(nz_features) < 3:
            nz_features = ranking_df['feature'].head(top_n).tolist()

        idx_keep = [feature_names.index(f) for f in nz_features
                    if f in feature_names]
        sv_sub  = shap_values[:, idx_keep]
        X_sub   = X_t[:, idx_keep] if hasattr(X_t, '__getitem__') else X_t
        names_sub = [feature_names[i] for i in idx_keep]

        fig = plt.figure(figsize=(9, max(5, 0.36 * len(idx_keep))))
        shap.summary_plot(sv_sub, X_sub, feature_names=names_sub,
                          max_display=len(idx_keep), show=False)
        # titulo manual
        ax = plt.gca()
        ax.set_title(f'SHAP summary — {model_name}  '
                     f'({len(idx_keep)} features con |SHAP|>0)',
                     fontsize=11, pad=12)
        _save_fig(plt.gcf(), FIG_SHAP / f"{file_prefix}shap_summary_{_slug(model_name)}.{FIGURE_FORMAT}")
    except Exception as exc:
        warnings.warn(f"[SHAP] summary_plot fallo para {model_name}: {exc}")
        plt.close('all')
