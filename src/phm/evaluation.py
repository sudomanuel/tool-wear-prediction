"""
evaluation.py — metricas y rutinas de evaluacion.
"""
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def safe_to_csv(df: pd.DataFrame, path, retries: int = 3, wait: float = 1.0):
    """
    df.to_csv con reintentos. Si el archivo esta bloqueado por Excel u
    otro proceso, espera y reintenta. Como ultimo recurso escribe a un
    nombre alternativo con sufijo .new.csv y emite un warning.
    """
    path = Path(path)
    last_err = None
    for _ in range(retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            return path
        except PermissionError as exc:
            last_err = exc
            time.sleep(wait)
    fallback = path.with_suffix('.new.csv')
    df.to_csv(fallback, index=False)
    warnings.warn(
        f"[IO] {path.name} esta bloqueado (probable Excel abierto): "
        f"escribi {fallback.name}. Cierra el visor y renombra manualmente o re-ejecuta."
    )
    return fallback


def read_latest_csv(path) -> pd.DataFrame:
    """
    Lee path o path.new.csv (lo que sea mas reciente). Util cuando el
    archivo original quedo bloqueado en una corrida previa.
    """
    path = Path(path)
    new = path.with_suffix('.new.csv')
    candidates = [p for p in (path, new) if p.exists()]
    if not candidates:
        return pd.DataFrame()
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(chosen)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if len(np.unique(y_true)) > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = float('nan')
    # MAPE seguro: ignora elementos con y_true == 0
    mask = np.abs(y_true) > 1e-9
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float('nan')
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE_%': mape}


def evaluate_holdout(models: dict,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray,  y_test: np.ndarray) -> pd.DataFrame:
    """
    Entrena cada modelo en (X_train, y_train) y evalua en (X_test, y_test).
    Devuelve DataFrame con metricas por modelo.
    """
    rows = []
    for name, pipe in models.items():
        if pipe is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = clone(pipe)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
            mets = compute_metrics(y_test, y_pred)
        except Exception as exc:
            warnings.warn(f"[EVAL] {name} fallo en hold-out: {exc}")
            mets = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan}
        rows.append({'model': name, **mets,
                     'n_train': int(len(y_train)), 'n_test': int(len(y_test))})
    return pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)


def evaluate_loeo(models: dict, df, X_cols, target_col, group_col) -> pd.DataFrame:
    """
    Para cada modelo, hace 10 folds LOEO y reporta metricas agregadas
    sobre las 10 predicciones (no promediadas por fold).
    """
    from .splitting import loeo_iter
    rows = []
    for name, pipe in models.items():
        if pipe is None:
            continue
        y_true_all = []
        y_pred_all = []
        ok = True
        for train_df, test_df in loeo_iter(df, group_col=group_col):
            X_tr = train_df[X_cols].values.astype(float)
            y_tr = train_df[target_col].values.astype(float)
            X_te = test_df[X_cols].values.astype(float)
            y_te = test_df[target_col].values.astype(float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = clone(pipe)
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_te)
            except Exception as exc:
                warnings.warn(f"[LOEO] {name} fold fallo: {exc}")
                ok = False
                break
            y_true_all.extend(y_te.tolist())
            y_pred_all.extend(y_pred.tolist())
        if ok and y_true_all:
            mets = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
        else:
            mets = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan}
        rows.append({'model': name, **mets, 'n_folds': len(y_true_all)})
    return pd.DataFrame(rows).sort_values('MAE').reset_index(drop=True)


def make_predictions_df(model_name: str,
                        experiment_ids,
                        y_true,
                        y_pred,
                        extra: dict = None) -> pd.DataFrame:
    """
    Construye un DataFrame estandar de predicciones por experimento.
    Columnas: model, experiment_id, VB_real, VB_pred, residual,
              absolute_error, percentage_error, [+ extras].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eids   = np.asarray(experiment_ids).astype(int)
    res    = y_pred - y_true
    pct    = np.where(np.abs(y_true) > 1e-9,
                      (y_pred - y_true) / y_true * 100.0, np.nan)
    rows = []
    for i in range(len(eids)):
        row = {
            'model': model_name,
            'experiment_id': int(eids[i]),
            'VB_real': float(y_true[i]),
            'VB_pred': float(y_pred[i]),
            'residual': float(res[i]),
            'absolute_error': float(abs(res[i])),
            'percentage_error': float(pct[i]) if not np.isnan(pct[i]) else float('nan'),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def build_final_ranking(df_holdout: pd.DataFrame,
                        df_loeo: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida hold-out y LOEO en una tabla unica con ranking por MAE_loeo
    (mas honesto que hold-out con n=2).
    """
    df = df_holdout[['model', 'MAE', 'RMSE', 'R2', 'MAPE_%']].rename(columns={
        'MAE': 'MAE_holdout', 'RMSE': 'RMSE_holdout',
        'R2': 'R2_holdout', 'MAPE_%': 'MAPE_holdout',
    }).merge(
        df_loeo[['model', 'MAE', 'RMSE', 'R2', 'MAPE_%']].rename(columns={
            'MAE': 'MAE_loeo', 'RMSE': 'RMSE_loeo',
            'R2': 'R2_loeo', 'MAPE_%': 'MAPE_loeo',
        }), on='model', how='outer',
    )
    return df.sort_values('MAE_loeo').reset_index(drop=True)
