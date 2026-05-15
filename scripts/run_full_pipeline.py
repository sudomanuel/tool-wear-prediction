#!/usr/bin/env python3
"""
run_full_pipeline.py — orquestador del pipeline completo.

Pasos (ordenados de raw -> interpretabilidad):

  0) audit_data.py                   - inventario, missing, plots de calidad
  1) build_dataset.py                - experiment_features + modeling_dataset + manifests
  2) train_baselines.py              - baselines hold-out + LOEO + predicciones + figs
  3) run_tuning.py                   - RandomizedSearchCV + GridSearchCV (XGB) + figs
  4) run_augmentation_experiment.py  - augmentation paralela + predicciones + figs
  5) evaluate_models.py              - ranking consolidado + figs features + leakage audit
  6) run_shap_analysis.py            - SHAP sobre top modelos LOEO (datos REALES)

Si un paso falla, se aborta.
"""
import sys
import time
import subprocess
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SCRIPTS_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

STEPS = [
    ("Paso 0/6 — audit_data",                   "audit_data.py"),
    ("Paso 1/6 — build_dataset",                "build_dataset.py"),
    ("Paso 2/6 — train_baselines",              "train_baselines.py"),
    ("Paso 3/6 — run_tuning",                   "run_tuning.py"),
    ("Paso 4/6 — run_augmentation_experiment",  "run_augmentation_experiment.py"),
    ("Paso 5/6 — evaluate_models",              "evaluate_models.py"),
    ("Paso 6/6 — run_shap_analysis",            "run_shap_analysis.py"),
]


def _purge_legacy_csv_names():
    """Borra CSVs cuyo nombre quedo obsoleto en este refactor."""
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    obsolete = [
        "tuning_cvresults_elasticnet.csv",
        "tuning_cvresults_lasso.csv",
        "tuning_cvresults_randomforest.csv",
        "tuning_cvresults_ridge.csv",
        "tuning_cvresults_svr.csv",
        "tuning_cvresults_xgboost.csv",
        "tuning_grid_xgboost.csv",
    ]
    for n in obsolete:
        p = metrics_dir / n
        if p.exists():
            p.unlink()
            print(f"[CLEAN] removed legacy: {p.name}")


def run_step(label: str, script_name: str) -> int:
    print("\n" + "#" * 70)
    print(f"# {label}")
    print("#" * 70)
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    dt = time.time() - t0
    print(f"\n[{label}] returncode={res.returncode}  ({dt:.1f}s)")
    return res.returncode


def main():
    _purge_legacy_csv_names()

    for label, script in STEPS:
        rc = run_step(label, script)
        if rc != 0:
            print(f"\n[ABORT] {label} fallo con codigo {rc}.")
            sys.exit(rc)

    # Resumen final
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import pandas as pd
    from phm.config import (
        METRICS_DIR, FIGURES_DIR, PROCESSED_DATASET, TARGET_COLUMN,
        PREDICTIONS_DIR, SPLITS_DIR,
    )

    df = pd.read_csv(PROCESSED_DATASET)
    df_ho = pd.read_csv(METRICS_DIR / "model_comparison_holdout.csv")
    df_lo = pd.read_csv(METRICS_DIR / "model_comparison_loeo.csv")
    df_aug = pd.read_csv(METRICS_DIR / "augmentation_comparison.csv")
    df_tune = pd.read_csv(METRICS_DIR / "tuning_results.csv") if (METRICS_DIR / "tuning_results.csv").exists() else pd.DataFrame()
    df_rank = pd.read_csv(METRICS_DIR / "final_model_ranking.csv")
    leaks = pd.read_csv(METRICS_DIR / "leakage_checks.csv")

    best_ho   = df_ho.sort_values('MAE').iloc[0]
    best_lo   = df_lo.sort_values('MAE').iloc[0]
    best_tune = df_tune.sort_values('MAE').iloc[0] if not df_tune.empty else None

    # Augmentation: hubo alguna estrategia con MAE menor que la del 'none'?
    aug_msg = "no mejora significativa respecto a sin augmentation"
    if not df_aug.empty:
        none_mae = (df_aug[df_aug['augmentation_strategy'] == 'none']
                    .groupby('model')['MAE'].first())
        other = df_aug[df_aug['augmentation_strategy'] != 'none']
        gains = []
        for _, row in other.iterrows():
            mname = row['model']
            base = none_mae.get(mname)
            if base is None or pd.isna(base) or pd.isna(row['MAE']):
                continue
            if row['MAE'] < base - 0.5:    # mejora de al menos 0.5 µm
                gains.append((mname, row['augmentation_strategy'],
                              float(base - row['MAE'])))
        if gains:
            gains.sort(key=lambda x: -x[2])
            aug_msg = "; ".join([f"{m}+{s}: -{d:.2f} µm MAE" for m, s, d in gains[:3]])

    # SHAP: que modelos fueron explicados
    shap_dir = METRICS_DIR / "shap"
    shap_files = sorted(shap_dir.glob("shap_feature_ranking_*.csv"))
    shap_models = [f.stem.replace("shap_feature_ranking_", "") for f in shap_files]

    n_features = sum(1 for c in df.columns if c not in
                     {'experiment_id', 'tool_id', 'experiment_order',
                      'end_of_life', 'is_augmented', TARGET_COLUMN}
                     and df[c].dtype != object)

    print("\n" + "=" * 70)
    print("Pipeline ejecutado correctamente.")
    print(f"Dataset procesado: {len(df)} experimentos.")
    print(f"Target: {TARGET_COLUMN}.")
    print(f"Features usadas: {n_features}.")
    print(f"Train/test split: {SPLITS_DIR / 'train_test_split.csv'}")
    print(f"Mejor modelo hold-out: {best_ho['model']}  "
          f"MAE={best_ho['MAE']:.2f}  R2={best_ho['R2']:.3f}")
    print(f"Mejor modelo LOEO   : {best_lo['model']}  "
          f"MAE={best_lo['MAE']:.2f}  R2={best_lo['R2']:.3f}")
    if best_tune is not None:
        print(f"Mejor modelo tuneado: {best_tune['model']}  "
              f"MAE={best_tune['MAE']:.2f}  R2={best_tune['R2']:.3f}")
    print(f"Augmentation: {aug_msg}")
    print(f"SHAP ejecutado para: {', '.join(shap_models) if shap_models else '(ninguno)'}")
    print(f"CSVs en:  {METRICS_DIR}/  y  {PREDICTIONS_DIR}/")
    print(f"Figures en: {FIGURES_DIR}/")
    # Leakage summary
    fails = leaks[leaks['status'] == 'FAIL']
    if fails.empty:
        print(f"Leakage checks: TODOS PASS ({len(leaks)} checks).")
    else:
        print(f"Leakage checks: {len(fails)} FAIL — ver leakage_checks.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
