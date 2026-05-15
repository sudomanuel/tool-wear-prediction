#!/usr/bin/env python3
"""
run_shap_analysis.py — paso 6 (FINAL) del pipeline.

Carga final_model_ranking.csv, selecciona top-2 LOEO + el mejor modelo
no-lineal disponible (si existe), carga los .joblib correspondientes y
calcula SHAP SOBRE DATOS REALES (10 filas, sin augmentation).

Background  = train real (hold-out 8/2)
Explicacion = todas las 10 filas reales (para ranking global)
              + filtro defensivo: si la columna is_augmented existe,
                se filtran las filas augmentadas.
"""
import sys
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
    METRICS_DIR, MODELS_DIR, ensure_output_dirs,
)
from phm.dataset_builder import get_feature_columns
from phm.splitting import load_holdout_split
from phm.shap_analysis import explain_model, SHAP_AVAILABLE


LINEAR_MODELS    = {'elasticnet', 'lasso', 'ridge'}
NONLINEAR_MODELS = {'svr', 'randomforest', 'xgboost', 'mlp',
                    'xgboost_random', 'xgboost_grid'}


def _slugify(name: str) -> str:
    """Normaliza nombres para localizar el .joblib."""
    n = name.lower().replace(' (tuned)', '_tuned').replace(' ', '_')
    return n


def _find_model_file(name_in_ranking: str) -> Path:
    """
    Dado un nombre como 'ElasticNet' o 'Ridge (tuned)' o 'XGBoost_random (tuned)'
    intenta resolver al .joblib correspondiente en outputs/models/.
    """
    n = name_in_ranking.lower()
    candidates = []
    if '(tuned)' in n:
        base = n.replace(' (tuned)', '').strip()
        candidates.append(MODELS_DIR / f"best_{base.replace(' ', '_')}_tuned.joblib")
    else:
        candidates.append(MODELS_DIR / f"{n.replace(' ', '_')}.joblib")
    for c in candidates:
        if c.exists():
            return c
    return None


def _normalize_ranking(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un df con columnas (model, MAE_loeo) ordenado por MAE_loeo.
    Soporta:
      - formato wide (con columna 'MAE_loeo')
      - formato long (con 'validation_type' = 'loeo' y columna 'MAE')
    """
    if 'MAE_loeo' in ranking_df.columns:
        out = ranking_df.dropna(subset=['MAE_loeo'])[['model', 'MAE_loeo']]
    elif 'validation_type' in ranking_df.columns:
        sub = ranking_df[ranking_df['validation_type'] == 'loeo'].copy()
        if 'pipeline_variant' in sub.columns:
            # quedarse solo con baselines: variant == 'baseline' (los tuneados
            # llevan '(tuned)' en model y son redundantes para SHAP).
            sub = sub[sub['pipeline_variant'] == 'baseline']
        out = sub.dropna(subset=['MAE'])[['model', 'MAE']].rename(columns={'MAE': 'MAE_loeo'})
    else:
        return pd.DataFrame(columns=['model', 'MAE_loeo'])
    return out.sort_values('MAE_loeo').reset_index(drop=True)


def select_models_for_shap(ranking_df: pd.DataFrame) -> list:
    """
    Devuelve una lista de (model_name_in_ranking, model_path) seleccionados:
      - top 2 por MAE_loeo
      - +1 mejor no-lineal aunque no este en el top 2
    Si el top 2 ya contiene un no-lineal, no se agrega tercero.
    """
    df = _normalize_ranking(ranking_df)
    if df.empty:
        return []
    chosen = []
    for _, row in df.head(2).iterrows():
        path = _find_model_file(row['model'])
        if path is None:
            warnings.warn(f"[SHAP] no encontre .joblib para {row['model']}, se omite")
            continue
        chosen.append((row['model'], path))

    chosen_names = {n.lower().split(' ')[0] for n, _ in chosen}
    if chosen_names & NONLINEAR_MODELS:
        return chosen
    for _, row in df.iterrows():
        base = row['model'].lower().split(' ')[0]
        if base in NONLINEAR_MODELS and (row['model'], None) not in [(n, None) for n, _ in chosen]:
            path = _find_model_file(row['model'])
            if path is not None:
                chosen.append((row['model'], path))
                break
    return chosen


def main():
    ensure_output_dirs()
    print("=" * 60)
    print("PASO 6 — SHAP analysis (modelos ya entrenados, datos REALES)")
    print("=" * 60)

    if not SHAP_AVAILABLE:
        print("[SHAP] shap no esta instalado. Se omite todo el paso.")
        return

    ranking_path = METRICS_DIR / "final_model_ranking.csv"
    if not ranking_path.exists():
        print(f"ERROR: falta {ranking_path}. Corre evaluate_models.py primero.")
        sys.exit(1)
    ranking_df = pd.read_csv(ranking_path)

    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}.")
        sys.exit(1)
    df = pd.read_csv(PROCESSED_DATASET)

    # Filtro defensivo: SHAP solo sobre filas reales.
    if 'is_augmented' in df.columns:
        n_before = len(df)
        df = df[df['is_augmented'] == False].reset_index(drop=True)
        if len(df) != n_before:
            print(f"[SHAP] filtre filas augmentadas: {n_before} -> {len(df)}")

    feat_cols = get_feature_columns(df)
    train_df, test_df = load_holdout_split(df, group_col=EXPERIMENT_ID_COL)

    X_train_real   = train_df[feat_cols].values.astype(float)
    X_explain_real = df[feat_cols].values.astype(float)
    explain_eids   = df[EXPERIMENT_ID_COL].astype(int).tolist()

    targets = select_models_for_shap(ranking_df)
    if not targets:
        print("[SHAP] ningun modelo seleccionable.")
        return

    print(f"[SHAP] modelos seleccionados: {[t[0] for t in targets]}")
    print(f"[SHAP] background (train real) shape: {X_train_real.shape}")
    print(f"[SHAP] datos a explicar (real)  shape: {X_explain_real.shape}")
    print(f"[SHAP] features: {len(feat_cols)}")

    for name, path in targets:
        print(f"\n--- {name}  ({path.name}) ---")
        try:
            pipe = joblib.load(path)
        except Exception as exc:
            warnings.warn(f"[SHAP] no se pudo cargar {path}: {exc}")
            continue
        res = explain_model(
            model_name=name,
            pipeline=pipe,
            X_train_real=X_train_real,
            X_explain_real=X_explain_real,
            feature_names=feat_cols,
            explain_experiment_ids=explain_eids,
            top_n=20,
        )
        if res is None:
            print(f"  [WARN] SHAP omitido para {name}")
        else:
            rdf, _ = res
            print(f"  top-5 features (|SHAP| medio):")
            print(rdf.head(5)[['feature', 'mean_abs_shap']].to_string(index=False))

    print(f"\n[OK] resultados en {METRICS_DIR / 'shap'}")
    print(f"[OK] figuras   en {Path('outputs/figures/shap')}")


if __name__ == "__main__":
    main()
