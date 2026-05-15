#!/usr/bin/env python3
"""
run_tuning.py — paso 3 del pipeline.

Tuning ligero (RandomizedSearchCV con GroupKFold) sobre el TRAIN del
hold-out. Para XGBoost refina con GridSearchCV pequeno (opcional).
Evalua tambien LOEO con el mejor estimador (refit-completo por fold).
"""
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm.config import (
    PROCESSED_DATASET, EXPERIMENT_ID_COL, TARGET_COLUMN,
    METRICS_DIR, MODELS_DIR, FIG_TUNING,
    ensure_output_dirs,
)
from phm.visualization import bar_metric_in
from phm.dataset_builder import get_feature_columns
from phm.splitting import load_holdout_split
from phm.modeling import (
    build_ridge, build_lasso, build_elasticnet, build_svr,
    build_rf, build_xgb, XGBOOST_AVAILABLE,
)
from phm.tuning import tune_model, refine_xgb_grid
from phm.evaluation import compute_metrics, evaluate_loeo


def main():
    ensure_output_dirs()
    print("=" * 60)
    print("PASO 3 — Tuning (RandomizedSearchCV + opcional GridSearchCV XGB)")
    print("=" * 60)

    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}. Corre build_dataset.py")
        sys.exit(1)

    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)

    train_df, test_df = load_holdout_split(df, group_col=EXPERIMENT_ID_COL)
    X_train = train_df[feat_cols].values.astype(float)
    y_train = train_df[TARGET_COLUMN].values.astype(float)
    groups  = train_df[EXPERIMENT_ID_COL].values
    X_test  = test_df[feat_cols].values.astype(float)
    y_test  = test_df[TARGET_COLUMN].values.astype(float)

    print(f"[INFO] train={len(train_df)}, test={len(test_df)}, features={len(feat_cols)}")

    candidates = {
        'Ridge':        build_ridge(),
        'Lasso':        build_lasso(),
        'ElasticNet':   build_elasticnet(),
        'SVR':          build_svr(),
        'RandomForest': build_rf(),
    }
    if XGBOOST_AVAILABLE:
        candidates['XGBoost'] = build_xgb()

    summary_rows = []
    best_estimators = {}
    best_params_all = {}

    for name, pipe in candidates.items():
        print(f"\n--- tuning {name} ---")
        try:
            best_est, best_params, cv_res = tune_model(
                name, pipe, X_train, y_train, groups,
                n_iter=20, cv_splits=5,
            )
        except Exception as exc:
            warnings.warn(f"[TUNE] {name} fallo: {exc}")
            continue

        if cv_res is not None:
            # nombre canonico nuevo
            cv_res.to_csv(METRICS_DIR / f"tuning_cv_results_{name.lower()}.csv",
                          index=False)

        # Refinamiento solo para XGBoost
        if name == 'XGBoost':
            try:
                refined_est, refined_params, grid_res = refine_xgb_grid(
                    best_params, pipe, X_train, y_train, groups,
                )
                grid_res.to_csv(METRICS_DIR / "tuning_cv_results_xgboost_grid.csv", index=False)
                # Y guardamos tambien el de Random con sufijo claro
                if cv_res is not None:
                    cv_res.to_csv(METRICS_DIR / "tuning_cv_results_xgboost_random.csv", index=False)
                # Si el grid es mejor (menor MAE CV), nos quedamos con el
                # Compararemos por MAE en hold-out abajo, asi que conservamos ambos
                best_estimators['XGBoost_random'] = best_est
                best_params_all['XGBoost_random'] = best_params
                best_estimators['XGBoost_grid']   = refined_est
                best_params_all['XGBoost_grid']   = refined_params
            except Exception as exc:
                warnings.warn(f"[TUNE] XGB grid refinement omitido: {exc}")
                best_estimators[name] = best_est
                best_params_all[name] = best_params
        else:
            best_estimators[name] = best_est
            best_params_all[name] = best_params

    # Evaluar cada mejor estimador en hold-out (test real)
    print("\n--- evaluacion hold-out (test real) ---")
    for name, est in best_estimators.items():
        try:
            y_pred = est.predict(X_test)
            mets = compute_metrics(y_test, y_pred)
        except Exception as exc:
            warnings.warn(f"[EVAL] {name} fallo: {exc}")
            mets = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan}
        summary_rows.append({
            'model': name + ' (tuned)',
            'tuning_method': 'GridSearchCV' if 'grid' in name.lower() else 'RandomizedSearchCV',
            'pipeline_variant': 'tuned',
            'validation_type': 'holdout',
            **mets,
            'best_params': json.dumps({k: _safe(v) for k, v in
                                       best_params_all.get(name, {}).items()}),
        })
        # Guardar modelo tuneado
        fname = f"best_{name.lower()}_tuned.joblib"
        try:
            joblib.dump(est, MODELS_DIR / fname)
        except Exception as exc:
            warnings.warn(f"[SAVE] {name}: {exc}")

    df_sum = pd.DataFrame(summary_rows).sort_values('MAE').reset_index(drop=True)
    df_sum.to_csv(METRICS_DIR / "tuning_results.csv", index=False)
    print("\n[TUNING — HOLDOUT]\n" + df_sum.to_string(index=False))

    # LOEO con los modelos tuneados
    loeo_inputs = {k + ' (tuned)': est for k, est in best_estimators.items()}
    df_loeo = evaluate_loeo(loeo_inputs, df, X_cols=feat_cols,
                            target_col=TARGET_COLUMN,
                            group_col=EXPERIMENT_ID_COL)
    df_loeo.to_csv(METRICS_DIR / "tuning_results_loeo.csv", index=False)
    print("\n[TUNING — LOEO]\n" + df_loeo.to_string(index=False))

    print(f"\n[OK] tuning_results.csv      -> {METRICS_DIR}")
    print(f"[OK] tuning_results_loeo.csv -> {METRICS_DIR}")

    # Figuras de tuning
    try:
        df_ho_plot = df_sum.dropna(subset=['MAE'])
        bar_metric_in(FIG_TUNING, df_ho_plot, 'MAE',  'MAE — Tuneados (hold-out)',  True,  'tuning_mae_comparison')
        bar_metric_in(FIG_TUNING, df_ho_plot, 'RMSE', 'RMSE — Tuneados (hold-out)', True,  'tuning_rmse_comparison')
        if df_ho_plot['R2'].notna().any():
            bar_metric_in(FIG_TUNING, df_ho_plot.dropna(subset=['R2']),
                          'R2', 'R2 — Tuneados (hold-out)', False, 'tuning_r2_comparison')

        # Comparacion XGB Random vs Grid: en vez de barras identicas en
        # hold-out (sin valor), graficamos la distribucion de MAE_CV
        # observada en los CV interiores.
        import matplotlib.pyplot as _plt
        rand_path = METRICS_DIR / "tuning_cv_results_xgboost_random.csv"
        grid_path = METRICS_DIR / "tuning_cv_results_xgboost_grid.csv"
        if rand_path.exists() and grid_path.exists():
            try:
                cv_r = pd.read_csv(rand_path)
                cv_g = pd.read_csv(grid_path)
                # mean_test_score viene en escala negativa (neg_MAE) -> a MAE positivo
                mae_r = -cv_r['mean_test_score'].dropna().values
                mae_g = -cv_g['mean_test_score'].dropna().values
                fig, axes = _plt.subplots(1, 2, figsize=(11, 4.5),
                                          gridspec_kw={'width_ratios': [1.0, 1.2]})
                # subplot 1: boxplot Random vs Grid
                bp = axes[0].boxplot([mae_r, mae_g], labels=['Random', 'Grid'],
                                     patch_artist=True)
                for patch, c in zip(bp['boxes'], ['#2E86AB', '#F18F01']):
                    patch.set_facecolor(c); patch.set_edgecolor('k')
                axes[0].set_ylabel('MAE_CV (µm)  — menor = mejor')
                axes[0].set_title(f'Distribucion CV  (n_random={len(mae_r)}, n_grid={len(mae_g)})')
                axes[0].grid(True, axis='y', alpha=0.3)
                axes[0].axhline(min(mae_r.min() if len(mae_r) else 0,
                                    mae_g.min() if len(mae_g) else 0),
                                color='red', ls=':', lw=1,
                                label=f"min CV = {min(mae_r.min() if len(mae_r) else 0, mae_g.min() if len(mae_g) else 0):.2f}")
                axes[0].legend(fontsize=8, loc='upper right')

                # subplot 2: top-5 configuraciones por MAE_CV (cada metodo)
                def _topk(df, k=5):
                    df = df.copy()
                    df['mae_cv'] = -df['mean_test_score']
                    return df.nsmallest(k, 'mae_cv')[['mae_cv']].reset_index(drop=True)

                topr = _topk(cv_r, 5); topg = _topk(cv_g, 5)
                ranks = np.arange(1, max(len(topr), len(topg)) + 1)
                w = 0.38
                axes[1].bar(ranks - w/2, topr['mae_cv'].reindex(range(len(ranks))),
                            width=w, label='Random', color='#2E86AB',
                            edgecolor='k', linewidth=0.4)
                axes[1].bar(ranks + w/2, topg['mae_cv'].reindex(range(len(ranks))),
                            width=w, label='Grid',  color='#F18F01',
                            edgecolor='k', linewidth=0.4)
                axes[1].set_xticks(ranks)
                axes[1].set_xticklabels([f"#{r}" for r in ranks])
                axes[1].set_xlabel('rank dentro del metodo')
                axes[1].set_ylabel('MAE_CV (µm)')
                axes[1].set_title('Top-5 configuraciones por MAE_CV')
                axes[1].grid(True, axis='y', alpha=0.3)
                axes[1].legend()
                fig.suptitle('XGBoost — Random vs Grid (búsqueda CV interna)', y=1.02)
                fig.tight_layout()
                out = FIG_TUNING / "xgboost_random_vs_grid.png"
                fig.savefig(out, dpi=130, bbox_inches='tight')
                _plt.close(fig)
            except Exception as exc:
                warnings.warn(f"[FIG] xgb random_vs_grid CV plot fallo: {exc}")

        print(f"[OK] figuras tuning:         {FIG_TUNING}")
    except Exception as exc:
        warnings.warn(f"[FIG] tuning fallaron: {exc}")


def _safe(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v) if v is not None else None


if __name__ == "__main__":
    main()
