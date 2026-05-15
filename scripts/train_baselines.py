#!/usr/bin/env python3
"""
train_baselines.py — paso 2 del pipeline.

- Carga el dataset procesado.
- Hace el split hold-out 8/2 (deterministico) y escribe el manifiesto.
- Escribe loeo_folds.csv.
- Entrena baselines sin tuning.
- Hold-out: guarda metricas, predicciones por experimento, modelos .joblib.
- LOEO: guarda metricas agregadas + predicciones por (fold, experimento).
- Plots de hold-out y LOEO en outputs/figures/holdout/ y outputs/figures/loeo/.
"""
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm.config import (
    PROCESSED_DATASET, EXPERIMENT_ID_COL, TARGET_COLUMN,
    METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR,
    FIG_HOLDOUT, FIG_LOEO, FIGURE_DPI, FIGURE_FORMAT,
    ensure_output_dirs,
)
from phm.dataset_builder import get_feature_columns
from phm.splitting import holdout_split, loeo_iter, write_loeo_folds
from phm.modeling import all_baseline_models
from phm.evaluation import compute_metrics, make_predictions_df
from phm.visualization import (
    bar_metric_in, actual_vs_predicted_in, residuals_plot_in,
    residuals_by_experiment_in,
)


def main():
    ensure_output_dirs()
    print("=" * 60)
    print("PASO 2 — Baselines (hold-out + LOEO)")
    print("=" * 60)

    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}. Corre build_dataset.py primero.")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)
    print(f"[INFO] features={len(feat_cols)}  filas={len(df)}")

    # Hold-out
    train_df, test_df = holdout_split(df, group_col=EXPERIMENT_ID_COL, save=True)
    write_loeo_folds(df, group_col=EXPERIMENT_ID_COL)
    print(f"[INFO] hold-out: train={len(train_df)}  test={len(test_df)}")
    print(f"[INFO] test experiment_ids: {sorted(test_df[EXPERIMENT_ID_COL].tolist())}")

    X_train = train_df[feat_cols].values.astype(float)
    y_train = train_df[TARGET_COLUMN].values.astype(float)
    X_test  = test_df[feat_cols].values.astype(float)
    y_test  = test_df[TARGET_COLUMN].values.astype(float)
    eids_test = test_df[EXPERIMENT_ID_COL].astype(int).values

    models = all_baseline_models()
    print(f"[INFO] modelos: {list(models.keys())}")

    # ---------- HOLDOUT ----------
    ho_rows = []
    pred_rows_holdout = []
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
            warnings.warn(f"[HO] {name} fallo: {exc}")
            mets = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan}
            y_pred = np.full_like(y_test, np.nan, dtype=float)

        ho_rows.append({'model': name, **mets,
                        'n_samples': int(len(y_test)),
                        'n_train': int(len(y_train)),
                        'n_test': int(len(y_test)),
                        'validation_type': 'holdout'})

        # predicciones por experimento (test)
        pdf = make_predictions_df(name, eids_test, y_test, y_pred,
                                  extra={'validation_type': 'holdout'})
        pred_rows_holdout.append(pdf)

        # guardar modelo
        try:
            joblib.dump(m, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        except Exception as exc:
            warnings.warn(f"[SAVE] {name}: {exc}")

    df_ho = pd.DataFrame(ho_rows).sort_values('MAE').reset_index(drop=True)
    df_ho.to_csv(METRICS_DIR / "model_comparison_holdout.csv", index=False)
    print("\n[HOLDOUT]\n" + df_ho.to_string(index=False))

    holdout_pred_df = pd.concat(pred_rows_holdout, ignore_index=True) if pred_rows_holdout else pd.DataFrame()
    holdout_pred_df.to_csv(PREDICTIONS_DIR / "holdout_predictions.csv", index=False)

    # ---------- LOEO ----------
    lo_rows = []
    pred_rows_loeo = []
    for name, pipe in models.items():
        if pipe is None:
            continue
        y_true_all, y_pred_all, eids_all, fold_all = [], [], [], []
        ok = True
        for fold_id, (tr_df, te_df) in enumerate(loeo_iter(df, group_col=EXPERIMENT_ID_COL), start=1):
            X_tr = tr_df[feat_cols].values.astype(float)
            y_tr = tr_df[TARGET_COLUMN].values.astype(float)
            X_te = te_df[feat_cols].values.astype(float)
            y_te = te_df[TARGET_COLUMN].values.astype(float)
            eid_te = te_df[EXPERIMENT_ID_COL].astype(int).values
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = clone(pipe)
                    m.fit(X_tr, y_tr)
                    y_p = m.predict(X_te)
            except Exception as exc:
                warnings.warn(f"[LOEO] {name} fold {fold_id} fallo: {exc}")
                ok = False
                break
            y_true_all.extend(y_te.tolist())
            y_pred_all.extend(y_p.tolist())
            eids_all.extend(eid_te.tolist())
            fold_all.extend([fold_id] * len(y_te))
        if ok and y_true_all:
            mets = compute_metrics(np.array(y_true_all), np.array(y_pred_all))
        else:
            mets = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan}
        lo_rows.append({'model': name, **mets,
                        'n_samples': int(len(y_true_all)),
                        'n_folds':   int(len(y_true_all)),
                        'validation_type': 'loeo'})
        # predicciones por (fold, experimento)
        pdf = make_predictions_df(name, eids_all, y_true_all, y_pred_all,
                                  extra={'validation_type': 'loeo'})
        pdf.insert(2, 'fold_id', fold_all)
        pred_rows_loeo.append(pdf)

    df_lo = pd.DataFrame(lo_rows).sort_values('MAE').reset_index(drop=True)
    df_lo.to_csv(METRICS_DIR / "model_comparison_loeo.csv", index=False)
    print("\n[LOEO]\n" + df_lo.to_string(index=False))

    loeo_pred_df = pd.concat(pred_rows_loeo, ignore_index=True) if pred_rows_loeo else pd.DataFrame()
    loeo_pred_df.to_csv(PREDICTIONS_DIR / "loeo_predictions.csv", index=False)

    # ---------- FIGURAS ----------
    # Holdout
    df_ho_plot = df_ho.dropna(subset=['MAE'])
    bar_metric_in(FIG_HOLDOUT, df_ho_plot, 'MAE',  'MAE — Hold-out',  True,  'mae_comparison')
    bar_metric_in(FIG_HOLDOUT, df_ho_plot, 'RMSE', 'RMSE — Hold-out', True,  'rmse_comparison')
    bar_metric_in(FIG_HOLDOUT.parent / 'holdout', df_ho_plot.dropna(subset=['R2']),
                  'R2', 'R2 — Hold-out', False, 'r2_comparison')
    # top-2 holdout: actual_vs_predicted + residuals
    top2_ho = df_ho_plot.head(2)['model'].tolist()
    pred_dict_ho = {}
    for n in top2_ho:
        sub = holdout_pred_df[holdout_pred_df['model'] == n]
        pred_dict_ho[n] = (sub['VB_real'].values, sub['VB_pred'].values)
    if pred_dict_ho:
        actual_vs_predicted_in(FIG_HOLDOUT, pred_dict_ho, 'Hold-out — Actual vs Predicted',
                               'actual_vs_predicted_best')
        residuals_plot_in(FIG_HOLDOUT, pred_dict_ho, 'Hold-out — Residuals',
                          'residuals_best')

    # LOEO
    df_lo_plot = df_lo.dropna(subset=['MAE'])
    bar_metric_in(FIG_LOEO, df_lo_plot, 'MAE',  'MAE — LOEO-CV',  True,  'mae_comparison')
    bar_metric_in(FIG_LOEO, df_lo_plot, 'RMSE', 'RMSE — LOEO-CV', True,  'rmse_comparison')
    bar_metric_in(FIG_LOEO, df_lo_plot.dropna(subset=['R2']),
                  'R2', 'R2 — LOEO-CV', False, 'r2_comparison')
    top2_lo = df_lo_plot.head(2)['model'].tolist()
    pred_dict_lo = {}
    for n in top2_lo:
        sub = loeo_pred_df[loeo_pred_df['model'] == n]
        pred_dict_lo[n] = (sub['VB_real'].values, sub['VB_pred'].values)
    if pred_dict_lo:
        actual_vs_predicted_in(FIG_LOEO, pred_dict_lo, 'LOEO-CV — Actual vs Predicted',
                               'actual_vs_predicted_best')
        residuals_plot_in(FIG_LOEO, pred_dict_lo, 'LOEO-CV — Residuals',
                          'residuals_best')
        residuals_by_experiment_in(FIG_LOEO, loeo_pred_df[loeo_pred_df['model'].isin(top2_lo)],
                                   'residuals_by_experiment')

    print(f"\n[OK] metrics:     {METRICS_DIR / 'model_comparison_holdout.csv'}")
    print(f"[OK] metrics:     {METRICS_DIR / 'model_comparison_loeo.csv'}")
    print(f"[OK] predictions: {PREDICTIONS_DIR / 'holdout_predictions.csv'}")
    print(f"[OK] predictions: {PREDICTIONS_DIR / 'loeo_predictions.csv'}")
    print(f"[OK] figures:     {FIG_HOLDOUT}/  {FIG_LOEO}/")


if __name__ == "__main__":
    main()
