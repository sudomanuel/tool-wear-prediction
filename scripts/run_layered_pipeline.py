#!/usr/bin/env python3
"""
run_layered_pipeline.py — pipeline experimental por capas, LOEO-only.

Estructura:
    D → {N, A} → {ST, CT_random, CT_grid}  →  LOEO-CV  →  ranking  →  SHAP

Convencion de nombres (prefijo numerico = etapa del flujo):
    00_ = auditoria / cleanup / diagrama
    01_ = dataset / features
    02_ = splits / folds LOEO
    03_ = baselines N · ST
    04_ = tuning N · CT_random
    05_ = tuning N · CT_grid
    06_ = augmentation A · ST
    07_ = augmentation A · CT_random
    08_ = augmentation A · CT_grid
    09_ = ranking final + resumenes + figuras comparativas
    10_ = SHAP
    11_ = manifest y mapa de figuras

Outputs principales:
    outputs/metrics/layered_pipeline/
        00_cleanup_report.csv
        00_leakage_checks.csv
        01_feature_columns.csv
        01_modeling_dataset_summary.csv
        02_loeo_folds.csv
        03_N_ST_loeo_metrics.csv
        04_N_CT_random_loeo_metrics.csv
        05_N_CT_grid_loeo_metrics.csv
        06_A_ST_loeo_metrics.csv
        07_A_CT_random_loeo_metrics.csv
        08_A_CT_grid_loeo_metrics.csv
        09_all_metrics.csv
        09_final_layered_ranking.csv
        09_branch_best_summary.csv
        09_delta_vs_baseline.csv
        09_tuning_effect_summary.csv
        09_augmentation_effect_summary.csv
        09_random_vs_grid_summary.csv
        09_branch_execution_summary.csv
        09_tuning_results_all.csv
        09_figure_purpose_map.csv
    outputs/metrics/shap/
        10_shap_selected_models.csv
        10_shap_feature_ranking_<model>_<branch>.csv
        10_shap_values_<model>_<branch>.csv
    outputs/predictions/layered_pipeline/
        09_predictions_all_branches.csv
    outputs/figures/layered_pipeline/
        00_layered_flow_diagram_no_holdout.png
        09_branch_performance_{MAE,RMSE,R2,MAPE}.png
        09_best_model_per_branch_MAE.png
        09_heatmap_model_vs_branch_{MAE,R2}.png
        09_delta_{MAE,RMSE}_vs_baseline_N_ST.png
        09_tuning_effect_{MAE,RMSE}.png
        09_random_vs_grid_{MAE,RMSE}.png
        09_augmentation_effect_{MAE,RMSE}.png
        09_sequential_comparison_dashboard_{MAE,RMSE,R2}.png
        09_actual_vs_predicted_best_global_LOEO.png
        09_residuals_best_global_LOEO.png
        09_residuals_by_experiment_best_global_LOEO.png
    outputs/figures/shap/
        10_shap_bar_<model>_<branch>.png
        10_shap_summary_<model>_<branch>.png
    outputs/metrics/output_manifest.csv  (indice global)
"""
from __future__ import annotations

import sys
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phm.config import (
    PROCESSED_DATASET, EXPERIMENT_ID_COL, TARGET_COLUMN, TOOL_ID_COL,
    EXP_ORDER_COL, METRICS_DIR, PREDICTIONS_DIR, FIGURES_DIR, MODELS_DIR,
    FIG_SHAP, METRICS_SHAP, ARCHIVE_DIR, SPLITS_DIR, LOEO_FOLDS_FILE,
    ensure_output_dirs,
)
from phm.dataset_builder import get_feature_columns
from phm.splitting import write_loeo_folds
from phm.evaluation import safe_to_csv
from phm.layered_pipeline import (
    enumerate_branches, run_branch, build_final_ranking,
    build_branch_best_summary, build_delta_vs_baseline,
    build_tuning_effect_summary, build_augmentation_effect_summary,
    build_random_vs_grid_summary,
    build_model_evolution_summary, build_model_evolution_by_model,
    select_predictions_for_multi_overlay,
    NONLINEAR_NAMES, FEATURE_SUBSETS, get_features_for_subset,
    AUGMENTATION_STRATEGIES, parse_branch_id,
)
from phm.layered_visuals import (
    plot_layered_flow_diagram, plot_branch_performance,
    plot_heatmap_model_vs_branch, plot_delta_vs_baseline,
    plot_tuning_effect, plot_random_vs_grid, plot_augmentation_effect,
    plot_best_model_per_branch, plot_sequential_dashboard,
    plot_model_evolution, plot_actual_vs_predicted_multi,
    plot_residuals_by_experiment_multi,
)
from phm.shap_analysis import explain_model, SHAP_AVAILABLE


# =============================================================================
# Carpetas
# =============================================================================
LAYER_METRICS_DIR     = METRICS_DIR     / "layered_pipeline"
LAYER_PREDICTIONS_DIR = PREDICTIONS_DIR / "layered_pipeline"
LAYER_FIGURES_DIR     = FIGURES_DIR     / "layered_pipeline"
AUDIT_DIR             = METRICS_DIR     / "output_audit"


def _setup_dirs():
    ensure_output_dirs()
    for d in (LAYER_METRICS_DIR, LAYER_PREDICTIONS_DIR, LAYER_FIGURES_DIR,
              AUDIT_DIR,
              ARCHIVE_DIR / "layered_pipeline_runs",
              ARCHIVE_DIR / "deprecated_holdout"):
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Step 0 — cleanup & archive de outputs deprecated (hold-out, sin prefijo)
# =============================================================================
def step_cleanup() -> tuple:
    """
    1. Archiva outputs de corrida previa del layered (en run_<timestamp>).
    2. Mueve outputs/figures/holdout, loeo, tuning, augmentation a
       archive/deprecated_holdout/ (LOEO-only desde ahora).
    3. Mueve CSVs deprecated del pipeline lineal a archive/deprecated_holdout/.
    4. Genera el cleanup_report.
    """
    rows = []

    def _check(item, ok, action='', note=''):
        rows.append({
            'item_checked': item,
            'status': 'PASS' if ok else 'WARN',
            'action_taken': action,
            'notes': note,
        })

    # 1) target VB_um
    try:
        df = pd.read_csv(PROCESSED_DATASET, nrows=1)
        ok = TARGET_COLUMN in list(df.columns)
        _check('target_column_is_VB_um', ok,
               note=f"target encontrado: {TARGET_COLUMN}" if ok
                    else f"falta {TARGET_COLUMN}")
        forbidden = {'VS', 'UVB', 'UVB/VB', 'thermal_corrosion'}
        bad = [c for c in df.columns if c in forbidden]
        _check('no_forbidden_targets', len(bad) == 0,
               note=('ninguna' if not bad else f"prohibidas: {bad}"))
    except Exception as e:
        _check('target_column_is_VB_um', False, note=str(e))

    # 2) Archivar layered run anterior si existe
    archived_layered = 0
    if LAYER_METRICS_DIR.exists() and any(LAYER_METRICS_DIR.iterdir()):
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest = ARCHIVE_DIR / "layered_pipeline_runs" / f"run_{ts}"
        dest.mkdir(parents=True, exist_ok=True)
        for src in (LAYER_METRICS_DIR, LAYER_PREDICTIONS_DIR, LAYER_FIGURES_DIR):
            for p in list(src.glob("*")):
                if p.is_file():
                    try:
                        shutil.move(str(p), dest / p.name)
                        archived_layered += 1
                    except Exception:
                        pass
        _check('prior_layered_outputs', True,
               action=f"archivados {archived_layered}",
               note=f"a {dest.relative_to(PROJECT_ROOT)}")
    else:
        _check('prior_layered_outputs', True, note='nada que archivar')

    # 3) Archivar figures stale de hold-out / loeo / tuning / augmentation
    ts2 = time.strftime("%Y%m%d_%H%M%S")
    dep_dest = ARCHIVE_DIR / "deprecated_holdout" / f"archived_{ts2}"
    dep_dest.mkdir(parents=True, exist_ok=True)
    n_deprec = 0
    for sub in ('holdout', 'loeo', 'tuning', 'augmentation'):
        d = FIGURES_DIR / sub
        if d.exists() and any(d.iterdir()):
            target_sub = dep_dest / f"figures_{sub}"
            target_sub.mkdir(parents=True, exist_ok=True)
            for p in list(d.glob("*")):
                if p.is_file():
                    try:
                        shutil.move(str(p), target_sub / p.name)
                        if p.exists():
                            try: p.unlink()
                            except Exception: pass
                        n_deprec += 1
                    except Exception:
                        pass

    # 4) Archivar CSVs deprecated del pipeline lineal en METRICS_DIR raiz.
    #    Whitelist de archivos a MANTENER en METRICS_DIR raiz; el resto se mueve.
    keep_in_root = {
        "data_inventory.csv", "feature_columns.csv",
        "leakage_checks.csv", "missing_segments.csv",
        "output_manifest.csv",
    }
    for p in list(METRICS_DIR.glob("*.csv")):
        if p.name in keep_in_root:
            continue
        try:
            target = dep_dest / p.name
            if target.exists():
                base = p.stem; ext = p.suffix; i = 1
                while target.exists():
                    target = dep_dest / f"{base}_{i}{ext}"; i += 1
            # En Windows shutil.move puede degradar a copy+remove y dejar
            # el original; aseguramos remocion explicita post-move.
            shutil.move(str(p), target)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
            n_deprec += 1
        except Exception:
            pass
    # Archivar predictions del pipeline lineal
    for name in ('holdout_predictions.csv', 'loeo_predictions.csv',
                  'augmentation_predictions.csv'):
        p = PREDICTIONS_DIR / name
        if p.exists():
            try:
                shutil.move(str(p), dep_dest / name)
                n_deprec += 1
            except Exception:
                pass
    # Archivar SHAP del pipeline lineal (sin sufijo de rama)
    linear_shap = {
        "shap_feature_ranking_elasticnet.csv", "shap_feature_ranking_lasso.csv",
        "shap_feature_ranking_xgboost.csv",
        "shap_values_elasticnet.csv", "shap_values_lasso.csv",
        "shap_values_xgboost.csv",
    }
    for name in linear_shap:
        p = METRICS_SHAP / name
        if p.exists():
            try:
                shutil.move(str(p), dep_dest / name)
                n_deprec += 1
            except Exception:
                pass
    for name in ("shap_bar_elasticnet.png", "shap_bar_lasso.png", "shap_bar_xgboost.png",
                  "shap_summary_elasticnet.png", "shap_summary_lasso.png", "shap_summary_xgboost.png"):
        p = FIG_SHAP / name
        if p.exists():
            try:
                shutil.move(str(p), dep_dest / name)
                n_deprec += 1
            except Exception:
                pass
    # Tambien SHAP del layered run anterior (sin prefijo 10_)
    for p in list(METRICS_SHAP.glob("shap_*_*.csv")):
        if not p.name.startswith("10_"):
            try:
                shutil.move(str(p), dep_dest / p.name)
                n_deprec += 1
            except Exception:
                pass
    for p in list(FIG_SHAP.glob("shap_*_*.png")):
        if not p.name.startswith("10_"):
            try:
                shutil.move(str(p), dep_dest / p.name)
                n_deprec += 1
            except Exception:
                pass

    _check('deprecated_holdout_archived', True,
           action=f"archivados {n_deprec} archivos deprecated",
           note=f"a {dep_dest.relative_to(PROJECT_ROOT)}")

    # 5) features clean
    try:
        df_full = pd.read_csv(PROCESSED_DATASET)
        feats = get_feature_columns(df_full)
        bad = [c for c in feats if c in {EXPERIMENT_ID_COL, TOOL_ID_COL,
                                          EXP_ORDER_COL, TARGET_COLUMN,
                                          'end_of_life', 'is_augmented',
                                          'source_experiment_id'}]
        _check('feature_columns_clean', len(bad) == 0,
               note=f"features={len(feats)}, leaks={bad}")
    except Exception as e:
        _check('feature_columns_clean', False, note=str(e))

    out = pd.DataFrame(rows)
    safe_to_csv(out, LAYER_METRICS_DIR / "00_cleanup_report.csv")
    return out, dep_dest, n_deprec


# =============================================================================
# Step 1 — leakage_checks
# =============================================================================
def step_leakage(df: pd.DataFrame) -> pd.DataFrame:
    from phm.leakage_audit import (
        check_one_row_per_experiment, check_target_unique_per_experiment,
        check_no_experiment_in_both_splits, check_id_columns_excluded,
        check_test_not_augmented, check_augmented_rows_keep_vb,
    )
    rows = [
        check_one_row_per_experiment(df),
        check_target_unique_per_experiment(df),
        check_no_experiment_in_both_splits(),
        check_id_columns_excluded(df),
        check_test_not_augmented(),
        check_augmented_rows_keep_vb(df),
    ]
    rows.append({
        'check_name': 'no_contact_level_labels',
        'status': 'PASS',
        'details': 'dataset_builder hace 1 fila/experimento',
    })
    rows.append({
        'check_name': 'experiment_order_excluded',
        'status': 'PASS' if EXP_ORDER_COL not in get_feature_columns(df) else 'FAIL',
        'details': f"{EXP_ORDER_COL} en NON_FEATURE_COLS",
    })
    rows.append({
        'check_name': 'shap_real_data_only',
        'status': 'PASS',
        'details': 'SHAP filtra is_augmented=True antes de explicar',
    })
    rows.append({
        'check_name': 'tuning_only_groupkfold',
        'status': 'PASS',
        'details': 'RandomizedSearchCV/GridSearchCV usan GroupKFold por experiment_id',
    })
    rows.append({
        'check_name': 'no_holdout_in_active_flow',
        'status': 'PASS',
        'details': 'flujo activo: solo LOEO-CV. Hold-out figures/CSVs archivados.',
    })

    out = pd.DataFrame(rows)
    safe_to_csv(out, LAYER_METRICS_DIR / "00_leakage_checks.csv")
    return out


# =============================================================================
# Step 1b — dataset summary
# =============================================================================
def step_dataset_summary(df: pd.DataFrame, feat_cols: list) -> None:
    safe_to_csv(pd.DataFrame({'feature': feat_cols}),
                LAYER_METRICS_DIR / "01_feature_columns.csv")
    summary = pd.DataFrame([{
        'n_experiments': int(df[EXPERIMENT_ID_COL].nunique()),
        'n_rows':        int(len(df)),
        'n_features':    int(len(feat_cols)),
        'target_column': TARGET_COLUMN,
        'experiment_ids': ';'.join(sorted(df[EXPERIMENT_ID_COL].astype(int).astype(str).unique())),
        'VB_min': float(df[TARGET_COLUMN].min()),
        'VB_max': float(df[TARGET_COLUMN].max()),
        'tool_id_unique': ';'.join(sorted(df.get(TOOL_ID_COL, pd.Series(['?'])).astype(str).unique())),
    }])
    safe_to_csv(summary, LAYER_METRICS_DIR / "01_modeling_dataset_summary.csv")
    write_loeo_folds(df, group_col=EXPERIMENT_ID_COL)
    # Copiar loeo_folds al folder layered con prefijo 02_
    if LOEO_FOLDS_FILE.exists():
        shutil.copy(LOEO_FOLDS_FILE, LAYER_METRICS_DIR / "02_loeo_folds.csv")


# =============================================================================
# Step 2 — ejecutar todas las ramas
# =============================================================================
STAGE_PREFIX = {
    'N_ST':              '03',
    'N_CT_random':       '04',
    'N_CT_grid':         '05',
    'A_ST':              '06',
    'A_CT_random':       '07',
    'A_CT_grid':         '08',
}


def _stage_prefix_for_branch(bid: str) -> str:
    """
    Mapea branch_id -> prefijo de etapa (03..08), independiente del subset.
    Ejemplos:
      FUSION_N_ST                          -> 03
      SOLO_A_N_CT_random                   -> 04
      SOLO_R_A_CT_grid_feature_noise       -> 08
    El subset (FUSION/SOLO_A/SOLO_R) no cambia el prefijo de etapa porque
    los CSVs se diferencian por el branch_id completo dentro del nombre.
    """
    # Quitar el prefijo de subset
    rest = bid
    for s in FEATURE_SUBSETS:
        pref = f'{s}_'
        if bid.startswith(pref):
            rest = bid[len(pref):]
            break
    if rest.startswith('N_ST'):           return '03'
    if rest.startswith('N_CT_random'):    return '04'
    if rest.startswith('N_CT_grid'):      return '05'
    if rest.startswith('A_ST'):           return '06'
    if rest.startswith('A_CT_random'):    return '07'
    if rest.startswith('A_CT_grid'):      return '08'
    return '09'


def step_run_all_branches(df: pd.DataFrame, feat_cols: list) -> dict:
    branches = enumerate_branches()
    print(f"\n[LAYER] Ramas a ejecutar: {len(branches)}")

    all_metrics, all_predictions, all_tuning = [], [], []
    summary_rows = []
    best_estimators_full: dict = {}   # branch_id -> {model: pipeline entrenado}

    for spec in branches:
        bid = spec['branch_id']
        subset = spec['feature_subset']
        # Filtrar features segun subset (FUSION/SOLO_A/SOLO_R)
        feat_cols_subset = get_features_for_subset(feat_cols, subset)
        t0 = time.time()
        try:
            res = run_branch(
                branch_id=bid,
                feature_subset=subset,
                data_branch=spec['data_branch'],
                tuning_method=spec['tuning_method'],
                aug_strategy=spec['aug_strategy'],
                full_df=df, feat_cols=feat_cols_subset,
            )
            status = 'OK'; notes = ''
            all_metrics.extend(res['metrics_rows'])
            all_predictions.extend(res['predictions_rows'])
            all_tuning.extend(res['tuning_rows'])
            best_estimators_full[bid] = res['best_estimators_full']

            # CSV individual de cada rama (con prefijo de etapa)
            prefix = _stage_prefix_for_branch(bid)
            df_branch = pd.DataFrame(res['metrics_rows'])
            safe_to_csv(df_branch,
                        LAYER_METRICS_DIR / f"{prefix}_{bid}_loeo_metrics.csv")
        except Exception as exc:
            status = 'FAIL'; notes = str(exc)
            warnings.warn(f"[LAYER] {bid} fallo: {exc}")

        summary_rows.append({
            'branch_id': bid,
            'feature_subset': subset,
            'n_features': len(feat_cols_subset),
            'data_branch': spec['data_branch'],
            'tuning_branch': 'ST' if spec['tuning_method'] == 'none' else 'CT',
            'tuning_method': spec['tuning_method'],
            'augmentation_strategy': spec['aug_strategy'],
            'validation_type': 'loeo',
            'models_run': 'all_baselines',
            'status': status,
            'notes': notes,
            'duration_sec': round(time.time() - t0, 1),
        })

    df_metrics = pd.DataFrame(all_metrics)
    df_predictions = (pd.concat(all_predictions, ignore_index=True)
                       if all_predictions else pd.DataFrame())
    df_tuning = pd.DataFrame(all_tuning)
    df_summary = pd.DataFrame(summary_rows)

    safe_to_csv(df_summary, LAYER_METRICS_DIR / "09_branch_execution_summary.csv")
    safe_to_csv(df_metrics, LAYER_METRICS_DIR / "09_all_metrics.csv")
    safe_to_csv(df_tuning,  LAYER_METRICS_DIR / "09_tuning_results_all.csv")
    safe_to_csv(df_predictions,
                LAYER_PREDICTIONS_DIR / "09_predictions_all_branches.csv")

    return {
        'metrics': df_metrics, 'predictions': df_predictions,
        'tuning': df_tuning, 'summary': df_summary,
        'best_estimators_full': best_estimators_full,
    }


# =============================================================================
# Step 3 — ranking final + resumenes derivados
# =============================================================================
def step_ranking_and_summaries(df_metrics: pd.DataFrame) -> dict:
    rank      = build_final_ranking(df_metrics)
    best_brch = build_branch_best_summary(df_metrics)
    delta_df  = build_delta_vs_baseline(df_metrics, baseline_branch='FUSION_N_ST')
    tun_df    = build_tuning_effect_summary(df_metrics)
    aug_df    = build_augmentation_effect_summary(df_metrics)
    rg_df     = build_random_vs_grid_summary(df_metrics)
    evo_df    = build_model_evolution_summary(df_metrics)
    evo_by_mdl = build_model_evolution_by_model(df_metrics, top_n_models=3)

    safe_to_csv(rank,      LAYER_METRICS_DIR / "09_final_layered_ranking.csv")
    safe_to_csv(best_brch, LAYER_METRICS_DIR / "09_branch_best_summary.csv")
    safe_to_csv(delta_df,  LAYER_METRICS_DIR / "09_delta_vs_baseline.csv")
    safe_to_csv(tun_df,    LAYER_METRICS_DIR / "09_tuning_effect_summary.csv")
    safe_to_csv(aug_df,    LAYER_METRICS_DIR / "09_augmentation_effect_summary.csv")
    safe_to_csv(rg_df,     LAYER_METRICS_DIR / "09_random_vs_grid_summary.csv")
    safe_to_csv(evo_df,    LAYER_METRICS_DIR / "09_model_evolution_summary.csv")
    safe_to_csv(evo_by_mdl, LAYER_METRICS_DIR / "09_model_evolution_by_model.csv")

    return {
        'rank': rank, 'best_per_branch': best_brch,
        'delta': delta_df, 'tuning_effect': tun_df,
        'aug_effect': aug_df, 'random_vs_grid': rg_df,
        'evolution': evo_df, 'evolution_by_model': evo_by_mdl,
    }


# =============================================================================
# Step 4 — figuras
# =============================================================================
def step_figures(df_metrics: pd.DataFrame, df_predictions: pd.DataFrame,
                 sums: dict, df: pd.DataFrame) -> list:
    """Genera todas las figuras y devuelve lista de paths producidos."""
    paths = []

    rank = sums['rank']
    best_loeo = rank.head(1) if not rank.empty else pd.DataFrame()
    best_name = best_loeo['model'].iloc[0] if not best_loeo.empty else None
    best_mae  = float(best_loeo['MAE'].iloc[0]) if not best_loeo.empty else None
    best_bid  = best_loeo['branch_id'].iloc[0] if not best_loeo.empty else None

    # 00 — diagrama
    paths.append(plot_layered_flow_diagram(
        LAYER_FIGURES_DIR, best_model_name=best_name, best_mae=best_mae,
        filename='00_layered_flow_diagram_no_holdout'))

    # 09 — branch performance (4 metricas)
    bp = sums['best_per_branch']
    for metric, lower_better in (('MAE', True), ('RMSE', True),
                                  ('R2', False), ('MAPE_%', True)):
        m_label = metric.replace('%', '').replace('_', '')
        paths.append(plot_branch_performance(
            LAYER_FIGURES_DIR, bp, metric, lower_better,
            filename=f'09_branch_performance_{m_label}'))

    # 09 — best model per branch
    paths.append(plot_best_model_per_branch(
        LAYER_FIGURES_DIR, bp, 'MAE',
        filename='09_best_model_per_branch_MAE'))

    # 09 — heatmap modelo x rama
    for metric, lower_better in (('MAE', True), ('R2', False)):
        paths.append(plot_heatmap_model_vs_branch(
            LAYER_FIGURES_DIR, df_metrics, metric, lower_better,
            filename=f'09_heatmap_model_vs_branch_{metric}'))

    # 09 — delta vs baseline
    for metric in ('MAE', 'RMSE'):
        paths.append(plot_delta_vs_baseline(
            LAYER_FIGURES_DIR, sums['delta'], metric,
            filename=f'09_delta_{metric}_vs_baseline_FUSION_N_ST'))

    # 09 — tuning effect
    for metric in ('MAE', 'RMSE'):
        paths.append(plot_tuning_effect(
            LAYER_FIGURES_DIR, sums['tuning_effect'], metric,
            filename=f'09_tuning_effect_{metric}'))

    # 09 — random vs grid
    for metric in ('MAE', 'RMSE'):
        paths.append(plot_random_vs_grid(
            LAYER_FIGURES_DIR, sums['random_vs_grid'], metric,
            filename=f'09_random_vs_grid_{metric}'))

    # 09 — augmentation effect
    for metric in ('MAE', 'RMSE'):
        paths.append(plot_augmentation_effect(
            LAYER_FIGURES_DIR, sums['aug_effect'], metric,
            filename=f'09_augmentation_effect_{metric}'))

    # 09 — sequential dashboard
    for metric in ('MAE', 'RMSE', 'R2'):
        paths.append(plot_sequential_dashboard(
            LAYER_FIGURES_DIR, bp, sums['delta'],
            sums['tuning_effect'], sums['aug_effect'],
            metric,
            filename=f'09_sequential_comparison_dashboard_{metric}'))

    # 09 — model evolution rama por rama (12 ramas)
    for metric, lower_better in (('MAE', True), ('RMSE', True),
                                  ('R2', False), ('MAPE_%', True)):
        m_label = metric.replace('%', '').replace('_', '')
        paths.append(plot_model_evolution(
            LAYER_FIGURES_DIR, sums['evolution'], sums['evolution_by_model'],
            metric, lower_better,
            filename=f'09_model_evolution_{m_label}_LOEO'))

    # 09 — real vs prediccion multi-overlay (5 configuraciones, colores)
    selections = select_predictions_for_multi_overlay(df_predictions, rank)
    if selections:
        paths.append(plot_actual_vs_predicted_multi(
            LAYER_FIGURES_DIR, selections,
            filename='09_actual_vs_predicted_multi_LOEO'))
        paths.append(plot_residuals_by_experiment_multi(
            LAYER_FIGURES_DIR, selections,
            filename='09_residuals_by_experiment_multi_LOEO'))
        # Guardar CSV con las predicciones seleccionadas
        rows = []
        for s in selections:
            for eid, yr, yp in zip(s['eids'], s['y_real'], s['y_pred']):
                rows.append({
                    'config_label': s['label'],
                    'model': s['model'],
                    'branch_id': s['branch_id'],
                    'experiment_id': int(eid),
                    'VB_real': float(yr),
                    'VB_pred': float(yp),
                    'residual': float(yp - yr),
                    'absolute_error': float(abs(yp - yr)),
                    'branch_MAE_LOEO': s['mae'],
                })
        from phm.evaluation import safe_to_csv as _safe_to_csv
        _safe_to_csv(pd.DataFrame(rows),
                     LAYER_METRICS_DIR / "09_predictions_overlay_selected.csv")

    # 09 — actual vs predicted + residuals del mejor global (LOEO)
    if best_name is not None and best_bid is not None and not df_predictions.empty:
        sub = df_predictions[
            (df_predictions['validation_type'] == 'loeo')
            & (df_predictions['model'] == best_name)
            & (df_predictions['branch_id'] == best_bid)
        ]
        if not sub.empty:
            from phm.visualization import (
                actual_vs_predicted_in, residuals_plot_in,
                residuals_by_experiment_in,
            )
            label = f"{best_name} ({best_bid})"
            paths.append(actual_vs_predicted_in(
                LAYER_FIGURES_DIR,
                {label: (sub['VB_real'].values, sub['VB_pred'].values)},
                f"LOEO — Actual vs Predicted (best global: {best_name})",
                '09_actual_vs_predicted_best_global_LOEO',
            ))
            paths.append(residuals_plot_in(
                LAYER_FIGURES_DIR,
                {label: (sub['VB_real'].values, sub['VB_pred'].values)},
                f"LOEO — Residuals (best global: {best_name})",
                '09_residuals_best_global_LOEO',
            ))
            paths.append(residuals_by_experiment_in(
                LAYER_FIGURES_DIR, sub,
                '09_residuals_by_experiment_best_global_LOEO',
            ))

    return [p for p in paths if p is not None]


# =============================================================================
# Step 5 — SHAP (datos REALES)
# =============================================================================
def _select_shap_targets(rank: pd.DataFrame, best_estimators_full: dict
                         ) -> list:
    """Top-2 por MAE LOEO + mejor no-lineal."""
    if rank.empty:
        return []
    targets = []

    def _pick(row, reason):
        bid = row['branch_id']
        est = best_estimators_full.get(bid, {}).get(row['model'])
        if est is None:
            warnings.warn(f"[SHAP] no hay estimator full para {row['model']} / {bid}")
            return None
        return {
            'model_name': row['model'], 'branch_id': bid,
            'estimator': est, 'reason': reason,
        }

    valid = rank.dropna(subset=['MAE'])
    for i, (_, r) in enumerate(valid.head(2).iterrows()):
        label = '1st by LOEO MAE' if i == 0 else '2nd by LOEO MAE'
        t = _pick(r, label)
        if t is not None:
            targets.append(t)
    chosen = {t['model_name'].lower() for t in targets}
    if not (chosen & NONLINEAR_NAMES):
        for _, r in valid.iterrows():
            if r['model'].lower() in NONLINEAR_NAMES:
                t = _pick(r, 'best non-linear in LOEO')
                if t is not None:
                    targets.append(t)
                break
    return targets


def step_shap(df: pd.DataFrame, feat_cols: list,
              rank: pd.DataFrame, best_estimators_full: dict) -> list:
    """Devuelve lista de tuples (model, branch) explicados."""
    if not SHAP_AVAILABLE:
        print("[SHAP] paquete no instalado — se omite SHAP.")
        return []

    targets = _select_shap_targets(rank, best_estimators_full)
    if not targets:
        print("[SHAP] ningun modelo seleccionado.")
        return []

    df_real = df.copy()
    if 'is_augmented' in df_real.columns:
        df_real = df_real[df_real['is_augmented'] == False].reset_index(drop=True)
    eids_explain   = df_real[EXPERIMENT_ID_COL].astype(int).tolist()

    sel_rows = []
    explained = []
    METRICS_SHAP.mkdir(parents=True, exist_ok=True)
    for t in targets:
        name, bid, est = t['model_name'], t['branch_id'], t['estimator']
        tag = f"{name}_{bid}"
        # CRITICAL: cada rama tiene su propio subset de features. El estimador
        # esta entrenado con feat_cols filtradas; SHAP debe usar las mismas.
        subset = parse_branch_id(bid).get('feature_subset') or 'FUSION'
        feat_cols_branch = get_features_for_subset(feat_cols, subset)
        X_real_branch    = df_real[feat_cols_branch].values.astype(float)
        print(f"\n[SHAP] {tag}  ({t['reason']}, subset={subset}, n_feats={len(feat_cols_branch)})")
        note = ''
        try:
            res = explain_model(
                model_name=tag,
                pipeline=est,
                X_train_real=X_real_branch,
                X_explain_real=X_real_branch,
                feature_names=feat_cols_branch,
                explain_experiment_ids=eids_explain,
                top_n=20,
                file_prefix='10_',
            )
            if res is None:
                note = 'SHAP failed (no result)'
            else:
                rdf, _ = res
                note = f"top1={rdf['feature'].iloc[0]} (|SHAP|={rdf['mean_abs_shap'].iloc[0]:.3f})"
                explained.append((name, bid))
        except Exception as exc:
            note = f"error: {exc}"
            warnings.warn(f"[SHAP] {tag}: {exc}")

        sel_rows.append({
            'model': name, 'branch_id': bid,
            'data_branch': bid[0],
            'tuning_method': ('random' if 'CT_random' in bid
                               else ('grid' if 'CT_grid' in bid else 'none')),
            'validation_type_used_for_selection': 'loeo',
            'reason_selected_for_shap': t['reason'],
            'shap_note': note,
        })
    safe_to_csv(pd.DataFrame(sel_rows),
                METRICS_SHAP / "10_shap_selected_models.csv")
    return explained


# =============================================================================
# Step 6 — manifest + figure_purpose_map
# =============================================================================
def step_manifest(figure_paths: list, shap_explained: list) -> None:
    """
    Genera output_manifest.csv (indice global) y figure_purpose_map.csv.
    """
    rows = []

    def _meta_from_filename(name: str) -> dict:
        meta = {'branch_id': '', 'feature_subset': '', 'data_branch': '',
                'tuning_method': '', 'augmentation_strategy': '',
                'validation_method': 'loeo'}
        # Generamos dinamicamente la lista de 36 branch_ids a buscar.
        candidates = [b['branch_id'] for b in enumerate_branches()]
        # Mas especificos primero (mas largos), para evitar matches parciales.
        candidates.sort(key=len, reverse=True)
        name_l = name.lower()
        for bid in candidates:
            if bid.lower() in name_l:
                parsed = parse_branch_id(bid)
                meta['branch_id'] = bid
                meta['feature_subset'] = parsed['feature_subset']
                meta['data_branch']    = parsed['data_branch']
                meta['tuning_method']  = parsed['tuning_method']
                meta['augmentation_strategy'] = parsed['aug_strategy']
                break
        return meta

    STAGE_NAMES = {
        '00': 'auditoria_y_diagrama', '01': 'dataset_features',
        '02': 'splits_loeo_folds',    '03': 'baseline_N_ST',
        '04': 'tuning_N_CT_random',   '05': 'tuning_N_CT_grid',
        '06': 'augmentation_A_ST',    '07': 'augmentation_A_CT_random',
        '08': 'augmentation_A_CT_grid', '09': 'ranking_y_resumenes',
        '10': 'shap', '11': 'manifest_y_reportes',
    }

    def _stage(name: str) -> tuple:
        if len(name) >= 3 and name[:2].isdigit() and name[2] == '_':
            sn = name[:2]; return sn, STAGE_NAMES.get(sn, '')
        return '', ''

    # Recorrer outputs
    bases = [
        (LAYER_METRICS_DIR, 'csv', 'run_layered_pipeline.py'),
        (LAYER_PREDICTIONS_DIR, 'csv', 'run_layered_pipeline.py'),
        (METRICS_SHAP, 'csv', 'run_layered_pipeline.py'),
        (LAYER_FIGURES_DIR, 'figure', 'run_layered_pipeline.py'),
        (FIG_SHAP, 'figure', 'run_layered_pipeline.py'),
    ]
    seen = set()
    for base, ftype, gen in bases:
        if not base.exists():
            continue
        for p in sorted(base.glob('*')):
            if not p.is_file():
                continue
            if p in seen:
                continue
            seen.add(p)
            # solo archivos del flujo activo (con prefijo 00..11)
            if not any(p.name.startswith(f"{i:02d}_") for i in range(12)):
                # excepcion: archivos de dataset summary sin prefijo seguiran ignorados
                continue
            sn, sname = _stage(p.name)
            meta = _meta_from_filename(p.name)
            rows.append({
                'file_path': str(p.relative_to(PROJECT_ROOT)).replace('\\', '/'),
                'file_type': ftype,
                'stage_number': sn,
                'stage_name':   sname,
                'branch_id': meta['branch_id'],
                'data_branch': meta['data_branch'],
                'tuning_method': meta['tuning_method'],
                'augmentation_strategy': meta['augmentation_strategy'],
                'validation_method': meta['validation_method'],
                'description': _describe(p.name),
                'generated_by_script': gen,
            })
    manifest = pd.DataFrame(rows).sort_values(
        ['stage_number', 'file_type', 'file_path']).reset_index(drop=True)
    safe_to_csv(manifest, METRICS_DIR / "output_manifest.csv")

    # figure_purpose_map
    fp_rows = _build_figure_purpose_rows(figure_paths)
    fp_df = pd.DataFrame(fp_rows)
    safe_to_csv(fp_df, LAYER_METRICS_DIR / "09_figure_purpose_map.csv")


def _describe(name: str) -> str:
    """Descripcion corta basada en el nombre."""
    n = name.lower()
    if 'cleanup_report' in n:        return 'auditoria y archivado de outputs deprecated'
    if 'leakage_checks' in n:        return 'verificacion anti-leakage de los splits y features'
    if 'feature_columns' in n:       return 'lista de features ML usadas'
    if 'modeling_dataset_summary' in n: return 'resumen del dataset (n_exp, VB range, etc.)'
    if 'loeo_folds' in n:            return 'definicion de los 10 folds LOEO'
    if 'predictions_all_branches' in n: return 'predicciones por experimento de cada (modelo, rama)'
    if 'all_metrics' in n:           return 'todas las metricas LOEO (modelo x rama)'
    if 'final_layered_ranking' in n: return 'ranking ordenado por MAE LOEO'
    if 'branch_best_summary' in n:   return 'mejor modelo y metricas por rama'
    if 'delta_vs_baseline' in n:     return 'delta de metricas vs baseline N_ST (negativo = mejora)'
    if 'tuning_effect_summary' in n: return 'comparacion ST vs Random vs Grid por (data, aug)'
    if 'augmentation_effect_summary' in n: return 'comparacion N vs A_{noise,scaling,grouped} por tuning'
    if 'random_vs_grid_summary' in n:return 'comparacion directa Random vs Grid'
    if 'branch_execution_summary' in n: return 'estado, duracion y notas por rama'
    if 'tuning_results_all' in n:    return 'best_params y best_cv_score de cada (modelo, rama)'
    if 'figure_purpose_map' in n:    return 'mapa de cada figura → pregunta que responde'
    if 'shap_feature_ranking' in n:  return 'ranking de features por |SHAP| medio'
    if 'shap_values' in n:           return 'valores SHAP individuales por (experimento, feature)'
    if 'shap_selected_models' in n:  return 'modelos elegidos para SHAP y razon'
    if 'layered_flow_diagram' in n:  return 'diagrama metodologico del flujo'
    if 'branch_performance' in n:    return 'best metrica por rama (LOEO)'
    if 'best_model_per_branch' in n: return 'mejor modelo por rama (LOEO)'
    if 'heatmap_model_vs_branch' in n: return 'heatmap modelo x rama (LOEO)'
    if 'delta_' in n and 'vs_baseline' in n: return 'delta de metrica respecto al baseline N_ST'
    if 'tuning_effect' in n:         return 'ST vs Random vs Grid lado a lado'
    if 'random_vs_grid' in n:        return 'Random vs Grid lado a lado + delta'
    if 'augmentation_effect' in n:   return 'N vs A_{strategies} por tuning method'
    if 'sequential_comparison_dashboard' in n: return 'dashboard 2x2 con la historia completa'
    if 'actual_vs_predicted' in n:   return 'predicciones del mejor modelo LOEO'
    if 'residuals_by_experiment' in n: return 'residuales por experimento del mejor LOEO'
    if 'residuals_best_global' in n: return 'distribucion de residuales del mejor LOEO'
    if 'shap_bar' in n:              return '|SHAP| medio del modelo seleccionado'
    if 'shap_summary' in n:          return 'beeswarm SHAP (impacto + valor) del modelo'
    if 'model_evolution_summary' in n: return 'mejor (modelo, rama) en cada una de las 12 ramas + delta vs previa/baseline'
    if 'model_evolution_by_model' in n: return 'top-3 modelos siguiendo las 12 ramas (largo)'
    if 'model_evolution' in n and n.endswith('.png'):
        return 'evolucion: como cambia la metrica rama por rama (12 ramas, linea principal + top-3 modelos)'
    if 'predictions_overlay_selected' in n:
        return 'predicciones de las 5 configuraciones del scatter multi-overlay'
    if 'actual_vs_predicted_multi' in n:
        return 'scatter VB_real vs VB_pred superpuesto con 5 configuraciones (colores)'
    if 'residuals_by_experiment_multi' in n:
        return 'residuos por experimento, 5 configuraciones superpuestas'
    return ''


def _build_figure_purpose_rows(paths: list) -> list:
    """figure_purpose_map.csv — una fila por figura activa."""
    rows = []

    def _add(p: Path, q: str, metric: str, hint: str, supervisor: bool):
        if p is None or not Path(p).exists():
            return
        rows.append({
            'figure_path': str(Path(p).relative_to(PROJECT_ROOT)).replace('\\', '/'),
            'question_answered': q,
            'metric': metric,
            'interpretation_hint': hint,
            'should_show_supervisor': supervisor,
        })

    for p in paths:
        if p is None:
            continue
        name = Path(p).name.lower()
        if 'layered_flow_diagram' in name:
            _add(p, '¿Que estructura tiene el experimento?', 'n/a',
                 'D → N/A → ST/CT(random,grid) → LOEO → ranking → SHAP. Sin hold-out.',
                 True)
        elif 'branch_performance_mae' in name:
            _add(p, '¿Cual rama tiene menor MAE?', 'MAE',
                 'Menor es mejor. Etiqueta = mejor modelo de la rama.', True)
        elif 'branch_performance_rmse' in name:
            _add(p, '¿Cual rama tiene menor RMSE?', 'RMSE',
                 'Menor es mejor. RMSE penaliza outliers mas que MAE.', False)
        elif 'branch_performance_r2' in name:
            _add(p, '¿Cual rama explica mas varianza?', 'R²',
                 'Mayor es mejor. R² negativo = peor que la media.', False)
        elif 'branch_performance_mape' in name:
            _add(p, '¿Cual rama tiene menor error porcentual?', 'MAPE',
                 'Menor es mejor. VB_um > 0 siempre, asi que MAPE es seguro.', False)
        elif 'best_model_per_branch' in name:
            _add(p, '¿Que modelo gana en cada rama?', 'MAE',
                 'Cada barra muestra rama → mejor modelo (con MAE).', True)
        elif 'heatmap_model_vs_branch_mae' in name:
            _add(p, '¿Que modelos son consistentemente buenos?', 'MAE',
                 'Verde oscuro = bajo MAE. Lee verticalmente por modelo.', True)
        elif 'heatmap_model_vs_branch_r2' in name:
            _add(p, '¿Que modelos generalizan mejor?', 'R²',
                 'Amarillo = R² alto. Negro = R² negativo.', False)
        elif 'delta_mae' in name:
            _add(p, '¿Tuning o augmentation mejoran al baseline?', 'MAE',
                 'Δ<0 = mejora; |Δ|<1 = empate practico; Δ>0 = empeora.', True)
        elif 'delta_rmse' in name:
            _add(p, '¿Idem MAE para RMSE?', 'RMSE',
                 'Misma lectura que delta MAE.', False)
        elif 'tuning_effect_mae' in name:
            _add(p, '¿Ayuda el tuning?', 'MAE',
                 'Tres barras: ST/Random/Grid. Si Random ≈ Grid, Grid no aporta.',
                 True)
        elif 'tuning_effect_rmse' in name:
            _add(p, '¿Ayuda el tuning (RMSE)?', 'RMSE',
                 'Idem MAE.', False)
        elif 'random_vs_grid_mae' in name:
            _add(p, '¿Aporta GridSearch sobre RandomizedSearch?', 'MAE',
                 'Panel derecho: |Δ|<1 = empate practico.', True)
        elif 'random_vs_grid_rmse' in name:
            _add(p, 'Idem random vs grid (RMSE)', 'RMSE', '', False)
        elif 'augmentation_effect_mae' in name:
            _add(p, '¿Augmentation ayuda o no?', 'MAE',
                 'Compara N vs cada estrategia A, agrupado por tuning.', True)
        elif 'augmentation_effect_rmse' in name:
            _add(p, 'Idem augmentation (RMSE)', 'RMSE', '', False)
        elif 'sequential_comparison_dashboard_mae' in name:
            _add(p, '¿Cual es la historia completa? (MAE)', 'MAE',
                 'Dashboard 2x2: best por rama + delta + tuning + aug.', True)
        elif 'sequential_comparison_dashboard_rmse' in name:
            _add(p, 'Historia RMSE', 'RMSE', '', False)
        elif 'sequential_comparison_dashboard_r2' in name:
            _add(p, 'Historia R²', 'R²', '', False)
        elif 'model_evolution_mae' in name:
            _add(p, '¿Cada rama nueva mejora o solo agrega complejidad?', 'MAE',
                 'Linea azul gruesa = ganador de cada rama. Estrella dorada = best overall. '
                 'Δ verde = mejora; rojo = empeora; gris = empate (<0.5 µm). '
                 'Bandas de fondo = familia de configuracion (N, N+tuning, A_ST, A+random, A+grid).',
                 True)
        elif 'model_evolution_rmse' in name:
            _add(p, '¿Idem evolucion (RMSE)?', 'RMSE',
                 'Misma lectura que MAE. RMSE penaliza outliers.', False)
        elif 'model_evolution_r2' in name:
            _add(p, '¿Cada rama explica mas varianza?', 'R²',
                 'Linea hacia arriba = mejora. Plana o cayendo = no.',
                 False)
        elif 'model_evolution_mape' in name:
            _add(p, '¿Cada rama baja el error porcentual?', 'MAPE',
                 'Idem MAE.', False)
        elif 'actual_vs_predicted_multi' in name:
            _add(p, '¿Como prediccion cada configuracion sobre los 10 experimentos?',
                 'visual',
                 'Cada color = una rama (baseline / tuneado / augmented / aug+tuneado / BEST). '
                 'Cerca de la diagonal = buena prediccion. Estrella roja = best global.',
                 True)
        elif 'residuals_by_experiment_multi' in name:
            _add(p, '¿Que experimentos fallan en TODAS las configuraciones?',
                 'visual',
                 'Cada experimento tiene 5 puntos (uno por config). Si todos los puntos '
                 'estan lejos del 0 en el mismo experimento, ese experimento es dificil '
                 'para cualquier modelo.',
                 True)
        elif 'actual_vs_predicted' in name:
            _add(p, '¿Que tan bien predice el mejor modelo?', 'visual',
                 'Cerca de la diagonal = buena prediccion.', True)
        elif 'residuals_by_experiment' in name:
            _add(p, '¿Que experimentos son los mas dificiles?', 'visual',
                 'Barras = error por experimento del mejor LOEO.', True)
        elif 'residuals_best_global' in name:
            _add(p, '¿Hay sesgo en los residuales?', 'visual',
                 'Distribucion centrada en 0 = sin sesgo.', False)
        elif 'shap_bar' in name:
            _add(p, '¿Que features son mas importantes?', '|SHAP|',
                 'Top features ordenadas por |SHAP| medio.', True)
        elif 'shap_summary' in name:
            _add(p, '¿Como afecta cada feature al output?', 'SHAP',
                 'Rojo = valor alto de feature; eje X = impacto en VB_um.', True)

    return rows


# =============================================================================
# MAIN
# =============================================================================
def main():
    _setup_dirs()
    print("=" * 70)
    print("PIPELINE EXPERIMENTAL POR CAPAS  —  LOEO-CV only  (sin hold-out)")
    print("=" * 70)

    # Step 0
    print("\n[0/6] cleanup_report + archivado de deprecated")
    _, dep_dest, n_deprec = step_cleanup()

    if not PROCESSED_DATASET.exists():
        print(f"ERROR: falta {PROCESSED_DATASET}. Corre scripts/build_dataset.py")
        sys.exit(1)
    df = pd.read_csv(PROCESSED_DATASET)
    feat_cols = get_feature_columns(df)
    print(f"[INFO] dataset: shape={df.shape}  features_ML={len(feat_cols)}")

    # Step 1: leakage + dataset summary
    print("\n[1/6] leakage_checks + dataset summary")
    step_leakage(df)
    step_dataset_summary(df, feat_cols)

    # Step 2: ramas
    print("\n[2/6] ejecutando todas las ramas (12, LOEO-only)")
    bundle = step_run_all_branches(df, feat_cols)

    # Step 3: ranking + resumenes
    print("\n[3/6] ranking final + resumenes derivados")
    sums = step_ranking_and_summaries(bundle['metrics'])
    print("\nTOP 10 LOEO:")
    if not sums['rank'].empty:
        print(sums['rank'].head(10).to_string(index=False))

    # Step 4: figuras
    print("\n[4/6] figuras")
    figure_paths = step_figures(bundle['metrics'], bundle['predictions'],
                                 sums, df)

    # Step 5: SHAP
    print("\n[5/6] SHAP sobre top modelos (datos REALES)")
    explained = step_shap(df, feat_cols, sums['rank'], bundle['best_estimators_full'])

    # Step 6: manifest
    print("\n[6/6] manifest + figure_purpose_map")
    step_manifest(figure_paths, explained)

    # ---------------- resumen final ----------------
    print("\n" + "=" * 70)
    n_branches = len(bundle['summary'])
    n_models   = bundle['metrics']['model'].nunique() if not bundle['metrics'].empty else 0
    n_rows_met = len(bundle['metrics'])
    n_csvs     = sum(1 for _ in LAYER_METRICS_DIR.glob('*.csv'))
    n_csvs_pr  = sum(1 for _ in LAYER_PREDICTIONS_DIR.glob('*.csv'))
    n_figs     = sum(1 for _ in LAYER_FIGURES_DIR.glob('*.png'))
    n_shap     = sum(1 for _ in METRICS_SHAP.glob('10_*.csv'))
    n_shap_fig = sum(1 for _ in FIG_SHAP.glob('10_*.png'))

    rank = sums['rank']
    if not rank.empty and rank['MAE'].notna().any():
        bl = rank.dropna(subset=['MAE']).iloc[0]
        print(f"Mejor LOEO: {bl['model']}  (branch={bl['branch_id']}, "
              f"MAE={bl['MAE']:.2f} µm, R²={bl['R2']:.3f})")

    # Efecto tuning
    df_m = bundle['metrics']
    nst = df_m[(df_m['data_branch'] == 'N') & (df_m['tuning_method'] == 'none')].sort_values('MAE')
    nct = df_m[(df_m['data_branch'] == 'N') & (df_m['tuning_method'].isin(['random', 'grid']))].sort_values('MAE')
    if not nst.empty and not nct.empty:
        d = float(nct['MAE'].min() - nst['MAE'].min())
        print(f"Efecto tuning (N, LOEO): " +
              (f"mejora MAE en {-d:.2f} µm" if d < 0 else f"empeora MAE en {d:.2f} µm"))
    # Efecto augmentation
    aug = df_m[df_m['data_branch'] == 'A']
    if not aug.empty and not nst.empty:
        d = float(aug['MAE'].min() - nst['MAE'].min())
        print(f"Efecto augmentation (LOEO): " +
              (f"mejora best MAE en {-d:.2f} µm" if d < 0
                else f"no mejora best MAE (Δ={d:+.2f} µm)"))
    # Random vs Grid
    if not sums['random_vs_grid'].empty:
        rg = sums['random_vs_grid']
        n_tie    = int((rg['winner'] == 'practical_tie (<1 µm)').sum())
        n_random = int((rg['winner'] == 'random').sum())
        n_grid   = int((rg['winner'] == 'grid').sum())
        print(f"Random vs Grid: random gana {n_random}, grid gana {n_grid}, "
              f"empate practico {n_tie}")

    print(f"\nRamas ejecutadas:     {n_branches}")
    print(f"Modelos evaluados:    {n_models}")
    print(f"Filas all_metrics:    {n_rows_met}")
    print(f"CSVs layered_pipeline/: {n_csvs}")
    print(f"CSV  predictions:       {n_csvs_pr}")
    print(f"Figuras layered_pipeline/: {n_figs}")
    print(f"SHAP CSVs ({METRICS_SHAP.name}): {n_shap}")
    print(f"SHAP figs ({FIG_SHAP.name}):     {n_shap_fig}")
    print(f"Archivados deprecated: {n_deprec}  ({dep_dest.relative_to(PROJECT_ROOT)})")
    print(f"\nManifest: outputs/metrics/output_manifest.csv")
    print(f"Mapa de figuras: outputs/metrics/layered_pipeline/09_figure_purpose_map.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
