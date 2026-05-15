#!/usr/bin/env python3
"""
run_augmentation_experiment.py — paso 4 del pipeline.

Compara baselines sin augmentation vs. con augmentation simple
(feature_noise, feature_scaling, grouped_scaling), aplicando la
augmentation SOLO al train del hold-out 8/2 y manteniendo el TEST
real e intacto en todos los casos.

Genera:
- data/interim/augmentation/train_augmented_<strategy>.csv
- outputs/metrics/augmentation_comparison.csv (con metadata enriquecida)
- outputs/predictions/augmentation_predictions.csv (por experimento de test)
- outputs/figures/augmentation/{mae,rmse,r2}_comparison.png
- outputs/figures/augmentation/actual_vs_predicted_best_augmented.png
- outputs/figures/augmentation/residuals_best_augmented.png
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import clone

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm.config import (
    PROCESSED_DATASET, EXPERIMENT_ID_COL, TARGET_COLUMN,
    METRICS_DIR, INTERIM_AUG_DIR, PREDICTIONS_DIR,
    FIG_AUGMENTATION,
    N_AUGMENTED_PER_EXPERIMENT, AUGMENTATION_NOISE_SIGMA,
    AUGMENTATION_SCALING_RANGE, ensure_output_dirs,
)
from phm.dataset_builder import get_feature_columns
from phm.splitting import load_holdout_split
from phm.modeling import all_baseline_models
from phm.evaluation import compute_metrics, make_predictions_df, safe_to_csv
from phm.augmentation import augment_train
from phm.visualization import (
    bar_metric_in, actual_vs_predicted_in, residuals_plot_in,
    grouped_bar_metric_in,
)


STRATEGIES = ['none', 'feature_noise', 'feature_scaling', 'grouped_scaling']


def main():
    ensure_output_dirs()
    print("=" * 60)
    print("PASO 4 — Augmentation experiment (paralelo, no reemplaza nada)")
    print("=" * 60)

    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}.")
        sys.exit(1)
    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)

    train_df, test_df = load_holdout_split(df, group_col=EXPERIMENT_ID_COL)
    print(f"[INFO] train={len(train_df)}  test={len(test_df)}")
    print(f"[INFO] test experiment_ids: {sorted(test_df[EXPERIMENT_ID_COL].tolist())}")

    X_test = test_df[feat_cols].values.astype(float)
    y_test = test_df[TARGET_COLUMN].values.astype(float)
    eids_test = test_df[EXPERIMENT_ID_COL].astype(int).values

    metric_rows = []
    pred_rows = []

    for strat in STRATEGIES:
        print(f"\n--- estrategia: {strat} ---")
        aug_df = augment_train(
            train_df, strategy=strat,
            n_augmented=N_AUGMENTED_PER_EXPERIMENT,
            noise_sigma=AUGMENTATION_NOISE_SIGMA,
            scaling_range=AUGMENTATION_SCALING_RANGE,
        )
        n_orig = int((~aug_df['is_augmented']).sum()) if 'is_augmented' in aug_df.columns else len(train_df)
        n_aug  = int(aug_df['is_augmented'].sum()) if 'is_augmented' in aug_df.columns else 0
        print(f"   train original: {n_orig}  | aumentadas: {n_aug}  | total: {len(aug_df)}")

        if strat != 'none':
            out_path = INTERIM_AUG_DIR / f"train_augmented_{strat}.csv"
            aug_df.to_csv(out_path, index=False)
            print(f"   guardado: {out_path}")

        X_tr = aug_df[feat_cols].values.astype(float)
        y_tr = aug_df[TARGET_COLUMN].values.astype(float)

        models = all_baseline_models()
        for name, pipe in models.items():
            if pipe is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = clone(pipe)
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_test)
                mets = compute_metrics(y_test, y_pred)
            except Exception as exc:
                warnings.warn(f"[AUG/{strat}/{name}] {exc}")
                mets = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE_%': np.nan}
                y_pred = np.full_like(y_test, np.nan, dtype=float)

            metric_rows.append({
                'model': name,
                'augmentation_strategy': strat,
                **mets,
                'n_train_original': n_orig,
                'n_train_after_augmentation': len(aug_df),
                'n_test': int(len(y_test)),
            })
            pdf = make_predictions_df(name, eids_test, y_test, y_pred,
                                      extra={'augmentation_strategy': strat})
            pred_rows.append(pdf)

    df_metrics = pd.DataFrame(metric_rows)
    p_metrics = safe_to_csv(df_metrics, METRICS_DIR / "augmentation_comparison.csv")
    df_pred = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    p_pred = safe_to_csv(df_pred, PREDICTIONS_DIR / "augmentation_predictions.csv")
    print(f"\n[OK] {p_metrics}")
    print(f"[OK] {p_pred}")

    # --- figures ---
    try:
        # Comparacion AGRUPADA por modelo, con una barra por estrategia.
        # Mucho mas legible que 32 barras planas.
        for metric, low_better in [('MAE', True), ('RMSE', True), ('R2', False)]:
            grouped_bar_metric_in(
                FIG_AUGMENTATION,
                df_metrics.dropna(subset=[metric]),
                group_col='model',
                sub_col='augmentation_strategy',
                metric=metric,
                title=f"{metric} por modelo — efecto de la augmentation",
                lower_is_better=low_better,
                filename=f"augmentation_{metric.lower()}_comparison",
            )

        # Adicionalmente, plot horizontal "delta MAE vs sin augmentation"
        # = cuanto cambia el MAE de cada modelo al aplicar cada estrategia.
        try:
            piv = df_metrics.pivot_table(index='model',
                                          columns='augmentation_strategy',
                                          values='MAE')
            if 'none' in piv.columns:
                delta = piv.subtract(piv['none'], axis=0).drop(columns='none')
                delta = delta.dropna(how='all')
                if not delta.empty:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    delta.plot(kind='barh', ax=ax, edgecolor='k', linewidth=0.4)
                    ax.axvline(0, color='k', lw=0.8)
                    ax.set_xlabel('Δ MAE vs baseline none (µm)  ←mejora | empeora→')
                    ax.set_title('Efecto neto de la augmentation por modelo')
                    ax.grid(True, axis='x', alpha=0.3)
                    ax.legend(title='strategy', fontsize=8)
                    fig.tight_layout()
                    out = FIG_AUGMENTATION / "augmentation_delta_mae.png"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out, dpi=130, bbox_inches='tight')
                    plt.close(fig)
        except Exception as exc:
            warnings.warn(f"[FIG] delta_mae fallo: {exc}")

        # mejores 2 (modelo, estrategia) por MAE — solo si son distintos
        best2 = df_metrics.dropna(subset=['MAE']).sort_values('MAE').head(3)
        pred_dict = {}
        for _, row in best2.iterrows():
            if len(pred_dict) >= 2:
                break
            key = f"{row['model']} / {row['augmentation_strategy']}"
            sub = df_pred[(df_pred['model'] == row['model']) &
                          (df_pred['augmentation_strategy'] == row['augmentation_strategy'])]
            pred_dict[key] = (sub['VB_real'].values, sub['VB_pred'].values)
        if pred_dict:
            actual_vs_predicted_in(FIG_AUGMENTATION, pred_dict,
                                   'Augmentation — Actual vs Predicted (best)',
                                   'actual_vs_predicted_best_augmented')
            residuals_plot_in(FIG_AUGMENTATION, pred_dict,
                              'Augmentation — Residuals (best)',
                              'residuals_best_augmented')
        print(f"[OK] figures:    {FIG_AUGMENTATION}")
    except Exception as exc:
        warnings.warn(f"[FIG] aug failed: {exc}")


if __name__ == "__main__":
    main()
