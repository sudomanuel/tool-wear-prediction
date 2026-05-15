#!/usr/bin/env python3
"""
evaluate_models.py — paso 5 del pipeline.

Consolida metricas de baselines + tuneados + augmentation en un ranking
unico. Genera figuras de features (correlacion + missingness) y los
plots resumen finales. Tambien corre el audit de leakage.
"""
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm.config import (
    PROCESSED_DATASET, EXPERIMENT_ID_COL, TARGET_COLUMN,
    METRICS_DIR, FIGURES_DIR, FIG_FEATURES,
    ensure_output_dirs,
)
from phm.dataset_builder import get_feature_columns
from phm.visualization import (
    plot_feature_missingness, plot_correlation_heatmap_top,
    plot_top_correlated_with_target,
)
from phm.leakage_audit import run_all_checks
from phm.evaluation import safe_to_csv, read_latest_csv


def _read_optional(path: Path) -> pd.DataFrame:
    return read_latest_csv(path)


def _normalize_metric_cols(df: pd.DataFrame, variant: str, vtype: str) -> pd.DataFrame:
    """Devuelve subset con (model, MAE, RMSE, R2, MAPE_%) + variant/vtype."""
    if df.empty:
        return df
    keep = ['model']
    for c in ('MAE', 'RMSE', 'R2', 'MAPE_%'):
        if c in df.columns:
            keep.append(c)
    out = df[keep].copy()
    out['pipeline_variant'] = variant
    out['validation_type']  = vtype
    return out


def build_consolidated_ranking() -> pd.DataFrame:
    """
    Une:
      - baseline holdout
      - baseline LOEO
      - tuned holdout
      - tuned LOEO
      - augmentation holdout
    En un solo CSV con columnas (model, pipeline_variant, validation_type,
    MAE, RMSE, R2, MAPE_%, rank_by_MAE, rank_by_RMSE, rank_by_R2).
    """
    ho_base = _normalize_metric_cols(_read_optional(METRICS_DIR / "model_comparison_holdout.csv"),
                                     'baseline', 'holdout')
    lo_base = _normalize_metric_cols(_read_optional(METRICS_DIR / "model_comparison_loeo.csv"),
                                     'baseline', 'loeo')

    tuned_ho = _read_optional(METRICS_DIR / "tuning_results.csv")
    if not tuned_ho.empty:
        tuned_ho = _normalize_metric_cols(tuned_ho, 'tuned', 'holdout')
    tuned_lo = _read_optional(METRICS_DIR / "tuning_results_loeo.csv")
    if not tuned_lo.empty:
        tuned_lo = _normalize_metric_cols(tuned_lo, 'tuned', 'loeo')

    aug = _read_optional(METRICS_DIR / "augmentation_comparison.csv")
    if not aug.empty:
        # incluimos solo filas !=  none (las none ya estan en baseline)
        if 'augmentation_strategy' in aug.columns:
            aug = aug[aug['augmentation_strategy'] != 'none'].copy()
            aug['pipeline_variant'] = 'augmented:' + aug['augmentation_strategy']
            aug['validation_type']  = 'holdout'
            keep = ['model', 'MAE', 'RMSE', 'R2', 'MAPE_%',
                    'pipeline_variant', 'validation_type']
            aug = aug[[c for c in keep if c in aug.columns]]

    parts = [df for df in (ho_base, lo_base, tuned_ho, tuned_lo, aug)
             if isinstance(df, pd.DataFrame) and not df.empty]
    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    # rankings (1 = mejor)
    df['rank_by_MAE']  = df['MAE'].rank(method='min', ascending=True)
    df['rank_by_RMSE'] = df['RMSE'].rank(method='min', ascending=True)
    df['rank_by_R2']   = df['R2'].rank(method='min', ascending=False)
    df = df.sort_values(['validation_type', 'pipeline_variant', 'MAE']).reset_index(drop=True)
    return df


def build_loeo_friendly_ranking(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Vista compacta para SHAP: solo modelos con LOEO disponible.
    Columnas estilo legacy: model, MAE_loeo, RMSE_loeo, R2_loeo.
    """
    sub = df_all[df_all['validation_type'] == 'loeo'].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.rename(columns={
        'MAE': 'MAE_loeo', 'RMSE': 'RMSE_loeo',
        'R2': 'R2_loeo', 'MAPE_%': 'MAPE_loeo',
    })
    sub = sub[['model', 'pipeline_variant', 'MAE_loeo', 'RMSE_loeo',
               'R2_loeo', 'MAPE_loeo']]
    return sub.sort_values('MAE_loeo').reset_index(drop=True)


def main():
    ensure_output_dirs()
    print("=" * 60)
    print("PASO 5 — Evaluacion final, ranking consolidado y figuras")
    print("=" * 60)

    # ranking consolidado
    df_all = build_consolidated_ranking()
    if df_all.empty:
        print("ERROR: no hay CSV de metricas para consolidar.")
        sys.exit(1)
    p_rank = safe_to_csv(df_all, METRICS_DIR / "final_model_ranking.csv")
    print(f"[OK] {p_rank}  ({len(df_all)} filas)")

    # Vista amigable para SHAP (mantiene compatibilidad con run_shap_analysis)
    df_loeo_view = build_loeo_friendly_ranking(df_all)
    if not df_loeo_view.empty:
        # No sobreescribo final_model_ranking; este es un view auxiliar
        safe_to_csv(df_loeo_view, METRICS_DIR / "final_model_ranking_loeo_view.csv")

    # Figuras de features
    if PROCESSED_DATASET.exists():
        df = pd.read_csv(PROCESSED_DATASET)
        feat_cols = get_feature_columns(df)
        try:
            plot_feature_missingness(FIG_FEATURES, df, feat_cols)
            plot_correlation_heatmap_top(FIG_FEATURES, df, feat_cols, TARGET_COLUMN, top_n=30)
            plot_top_correlated_with_target(FIG_FEATURES, df, feat_cols, TARGET_COLUMN, top_n=20)
            print(f"[OK] figuras features:  {FIG_FEATURES}")
        except Exception as exc:
            warnings.warn(f"[FIG] features fallaron: {exc}")

        # Leakage audit
        checks = run_all_checks(df)
        print("\n[LEAKAGE CHECKS]\n" + checks.to_string(index=False))

    # Resumen del ranking en consola
    print("\n[RANKING — top 10 por MAE]")
    print(df_all.sort_values('MAE').head(10).to_string(index=False))


if __name__ == "__main__":
    main()
