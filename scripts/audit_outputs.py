#!/usr/bin/env python3
"""
audit_outputs.py — audita figures y CSVs del proyecto.

Produce:
- outputs/metrics/output_audit/figures_audit.csv
- outputs/metrics/output_audit/figures_folder_summary.csv
- outputs/metrics/output_audit/csv_audit.csv
- outputs/metrics/output_audit/csv_folder_summary.csv

Clasifica cada archivo respecto al flujo actual (layered_pipeline sin
hold-out, solo LOEO-CV) y propone acciones (active/deprecated/archive/stale).
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURES_DIR     = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR     = PROJECT_ROOT / "outputs" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"
PROCESSED_DIR   = PROJECT_ROOT / "data"    / "processed"
INTERIM_DIR     = PROJECT_ROOT / "data"    / "interim"
AUDIT_DIR       = METRICS_DIR  / "output_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Clasificacion de figures
# -----------------------------------------------------------------------------
CURRENT_FIGURE_FOLDERS = {
    "layered_pipeline": ("active", "principal: 12 ramas LOEO-CV"),
    "shap":             ("active", "interpretabilidad, regenerada por layered"),
    "data_quality":     ("active", "audit_data (paso pre-flujo)"),
    "signals":          ("active", "audit_data (paso pre-flujo)"),
    "features":         ("supplemental", "exploracion de features, no parte del ranking"),
}
DEPRECATED_FIGURE_FOLDERS = {
    "holdout":     ("deprecated", "hold-out eliminado del flujo principal"),
    "loeo":        ("deprecated", "reemplazado por figures dentro de layered_pipeline"),
    "tuning":      ("deprecated", "tuning ahora vive dentro del flujo layered"),
    "augmentation":("deprecated", "augmentation ahora vive dentro del flujo layered"),
}

BRANCH_TOKENS = (
    "n_st", "n_ct_random", "n_ct_grid",
    "a_st_feature_noise", "a_ct_random_feature_noise", "a_ct_grid_feature_noise",
    "a_st_feature_scaling", "a_ct_random_feature_scaling", "a_ct_grid_feature_scaling",
    "a_st_grouped_scaling", "a_ct_random_grouped_scaling", "a_ct_grid_grouped_scaling",
)

def detect_branch(name_lower: str) -> str:
    for tok in BRANCH_TOKENS:
        if tok in name_lower:
            return tok.upper()
    return ""


def classify_figure(path: Path) -> dict:
    rel = path.relative_to(PROJECT_ROOT)
    parts = rel.parts
    name = path.name
    name_lower = name.lower()
    folder = parts[2] if len(parts) >= 3 else ""
    sub2   = parts[3] if len(parts) >= 4 else ""
    folder_key = folder

    belongs = False
    status  = "stale"
    stage   = ""
    note    = ""

    if folder in CURRENT_FIGURE_FOLDERS:
        st, msg = CURRENT_FIGURE_FOLDERS[folder]
        status = "active" if st == "active" else "supplemental"
        belongs = True
        note = msg
        if folder == "layered_pipeline":
            stage = "ranking_or_diagnostic"
            if "flow_diagram" in name_lower:
                stage = "00_flow_diagram"
            elif "_by_branch" in name_lower or "best_model_per_branch" in name_lower:
                stage = "09_ranking_visuals"
            elif "actual_vs_predicted" in name_lower or "residuals" in name_lower:
                stage = "09_best_global_diagnostic"
            elif "sequential" in name_lower:
                stage = "09_dashboard"
            elif "heatmap" in name_lower:
                stage = "09_heatmap"
            elif "delta" in name_lower:
                stage = "09_delta"
            elif "tuning_effect" in name_lower:
                stage = "09_tuning_effect"
            elif "augmentation_effect" in name_lower or "augmentation_comparison" in name_lower:
                stage = "09_augmentation_effect"
            elif "random_vs_grid" in name_lower:
                stage = "09_random_vs_grid"
            elif "branch_performance" in name_lower:
                stage = "09_branch_performance"
        elif folder == "shap":
            stage = "10_shap"
        elif folder == "data_quality":
            stage = "00_audit"
        elif folder == "signals":
            stage = "00_audit"
        elif folder == "features":
            stage = "01_features_eda"
    elif folder in DEPRECATED_FIGURE_FOLDERS:
        st, msg = DEPRECATED_FIGURE_FOLDERS[folder]
        status = st
        belongs = False
        note = msg
        stage = "legacy_holdout_or_split"

    branch = detect_branch(name_lower)

    # Reglas adicionales
    if folder == "layered_pipeline" and not name_lower.startswith(
        ("00_", "01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_", "11_")
    ):
        note = (note + " — sin prefijo numerico (sera regenerado).").strip()
        status = "stale"

    if folder == "shap" and not name_lower.startswith("10_"):
        # Determinar si tiene branch suffix (layered) o no (linear)
        if branch:
            note = "shap con branch (layered) — sera regenerado con prefijo 10_"
            status = "stale"
        else:
            note = "shap del pipeline lineal (sin branch) — deprecated en favor del layered"
            status = "deprecated"
            belongs = False

    return {
        "file_path": str(rel).replace("\\", "/"),
        "file_name": name,
        "folder": folder + (f"/{sub2}" if sub2 else ""),
        "belongs_to_current_pipeline": belongs,
        "pipeline_stage": stage,
        "branch_id_if_applicable": branch,
        "status": status,
        "action_taken": "",
        "notes": note,
    }


# -----------------------------------------------------------------------------
# Clasificacion de CSVs
# -----------------------------------------------------------------------------
EXPECTED_COLS_BY_FILENAME = {
    "all_metrics.csv": [
        "model", "data_branch", "tuning_method", "validation_type",
        "augmentation_strategy", "branch_id", "MAE", "RMSE", "R2",
    ],
    "predictions_all_branches.csv": [
        "model", "experiment_id", "VB_real", "VB_pred",
        "validation_type", "branch_id",
    ],
    "final_layered_ranking.csv": [
        "rank", "model", "MAE", "RMSE", "R2", "branch_id", "validation_type",
    ],
    "tuning_results_all.csv": [
        "model", "data_branch", "tuning_method", "branch_id", "best_params",
    ],
}

DEPRECATED_CSV_NAMES = {
    "model_comparison_holdout.csv":     ("deprecated", "hold-out fuera del flujo principal"),
    "holdout_predictions.csv":          ("deprecated", "hold-out fuera del flujo principal"),
    "final_model_ranking.csv":          ("deprecated", "reemplazado por final_layered_ranking"),
    "final_model_ranking_loeo_view.csv":("deprecated", "reemplazado por final_layered_ranking"),
    "augmentation_comparison.csv":      ("deprecated", "reemplazado por augmentation_effect_summary"),
    "augmentation_comparison.new.csv":  ("deprecated", "archivo .new.csv de re-intento"),
    "augmentation_predictions.csv":     ("deprecated", "reemplazado por predictions_all_branches"),
    "tuning_results.csv":               ("deprecated", "reemplazado por tuning_results_all"),
    "tuning_results_loeo.csv":          ("deprecated", "reemplazado por tuning_results_all"),
    "tuning_cv_results_elasticnet.csv": ("supplemental", "detalle CV — opcional para anexo"),
    "tuning_cv_results_lasso.csv":      ("supplemental", "detalle CV — opcional para anexo"),
    "tuning_cv_results_randomforest.csv":("supplemental","detalle CV — opcional para anexo"),
    "tuning_cv_results_ridge.csv":      ("supplemental", "detalle CV — opcional para anexo"),
    "tuning_cv_results_svr.csv":        ("supplemental", "detalle CV — opcional para anexo"),
    "tuning_cv_results_xgboost.csv":    ("supplemental", "detalle CV — opcional para anexo"),
    "tuning_cv_results_xgboost_grid.csv":("supplemental","detalle CV — opcional para anexo"),
    "tuning_cv_results_xgboost_random.csv":("supplemental","detalle CV — opcional para anexo"),
    "model_comparison_loeo.csv":        ("deprecated", "reemplazado por all_metrics + ranking"),
    "loeo_predictions.csv":             ("deprecated", "reemplazado por predictions_all_branches"),
    "data_inventory.csv":               ("active", "audit step (mantener)"),
    "missing_segments.csv":             ("active", "audit step (mantener)"),
    "feature_columns.csv":              ("active", "lista de features (mantener)"),
    "leakage_checks.csv":               ("active", "audit (sera regenerado dentro de layered)"),
}


def classify_csv(path: Path) -> dict:
    rel = path.relative_to(PROJECT_ROOT)
    parts = rel.parts
    name = path.name
    name_lower = name.lower()
    folder = "/".join(parts[1:-1])
    branch = detect_branch(name_lower)
    stage = ""

    belongs = False
    status  = "stale"
    note    = ""

    # SHAP CSVs
    if "shap" in folder:
        # 10_shap_selected_models.csv es el indice general (sin branch)
        if name_lower.startswith("10_"):
            belongs = True
            status = "active"
            note = "shap layered con prefijo 10_"
            stage = "10_shap"
        elif branch:
            belongs = True
            status = "stale"
            note = "shap layered sin prefijo 10_ (sera regenerado)"
            stage = "10_shap"
        else:
            belongs = False
            status = "deprecated"
            note = "shap pipeline lineal (sin branch)"
            stage = "10_shap"
    elif "layered_pipeline" in folder:
        belongs = True
        status = "active" if name_lower.startswith(
            ("00_","01_","02_","03_","04_","05_","06_","07_","08_","09_","10_","11_")
        ) else "stale"
        note = "" if status == "active" else "layered output sin prefijo numerico (sera regenerado)"
        stage = "layered"
    elif name == "output_manifest.csv":
        belongs = True
        status = "active"
        note = "indice global de outputs del flujo activo"
        stage = "11_manifest"
    elif name in DEPRECATED_CSV_NAMES:
        st, msg = DEPRECATED_CSV_NAMES[name]
        status = st
        note = msg
        belongs = (st == "active")
        stage = "linear_pipeline_or_audit"
    elif "processed" in folder.replace("\\", "/"):
        belongs = True
        status = "active"
        note = "dataset procesado (entrada del pipeline)"
        stage = "01_dataset"
    elif "interim" in folder.replace("\\", "/"):
        belongs = True
        status = "active"
        note = "intermediate (entrada del pipeline)"
        stage = "01_intermediate"
    elif folder.startswith("outputs/splits"):
        belongs = True
        status = "active"
        note = "split file (mantener)"
        stage = "02_splits"
    elif folder.startswith("outputs/predictions") and "layered" not in folder:
        belongs = False
        status = "deprecated"
        note = "predictions del pipeline lineal — reemplazado por layered/predictions_all_branches"
        stage = "deprecated_linear"

    # Columnas presentes vs esperadas
    expected = EXPECTED_COLS_BY_FILENAME.get(name, [])
    cols_present = True
    missing = []
    if expected and path.exists():
        try:
            df_head = pd.read_csv(path, nrows=1)
            missing = [c for c in expected if c not in df_head.columns]
            cols_present = (len(missing) == 0)
        except Exception as exc:
            cols_present = False
            missing = [f"(no se pudo leer: {exc})"]

    return {
        "file_path": str(rel).replace("\\", "/"),
        "file_name": name,
        "folder": folder,
        "belongs_to_current_pipeline": belongs,
        "pipeline_stage": stage,
        "expected_columns_present": cols_present,
        "missing_columns": ";".join(missing) if missing else "",
        "status": status,
        "action_taken": "",
        "notes": note,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Figures
    fig_rows = []
    if FIGURES_DIR.exists():
        for p in sorted(FIGURES_DIR.rglob("*.png")):
            fig_rows.append(classify_figure(p))
    fig_df = pd.DataFrame(fig_rows)
    fig_df.to_csv(AUDIT_DIR / "figures_audit.csv", index=False)

    # Figures folder summary
    if not fig_df.empty:
        agg = fig_df.groupby("folder").agg(
            n_files=("file_name", "count"),
            active_files=("status", lambda s: (s == "active").sum()),
            deprecated_files=("status", lambda s: (s == "deprecated").sum()),
            stale_files=("status", lambda s: (s == "stale").sum()),
            supplemental_files=("status", lambda s: (s == "supplemental").sum()),
        ).reset_index()
        agg["notes"] = agg["folder"].map(
            lambda f: CURRENT_FIGURE_FOLDERS.get(f.split("/")[0], DEPRECATED_FIGURE_FOLDERS.get(
                f.split("/")[0], ("unknown", "")))[1]
        )
        agg.to_csv(AUDIT_DIR / "figures_folder_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["folder", "n_files"]).to_csv(
            AUDIT_DIR / "figures_folder_summary.csv", index=False)

    # CSVs
    csv_rows = []
    for base in (METRICS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, INTERIM_DIR):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.csv")):
            # excluir output_audit/* para no contaminar
            if "output_audit" in p.parts:
                continue
            csv_rows.append(classify_csv(p))
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(AUDIT_DIR / "csv_audit.csv", index=False)

    if not csv_df.empty:
        agg = csv_df.groupby("folder").agg(
            n_files=("file_name", "count"),
            active_files=("status", lambda s: (s == "active").sum()),
            deprecated_files=("status", lambda s: (s == "deprecated").sum()),
            stale_files=("status", lambda s: (s == "stale").sum()),
            supplemental_files=("status", lambda s: (s == "supplemental").sum()),
        ).reset_index()
        agg["notes"] = ""
        agg.to_csv(AUDIT_DIR / "csv_folder_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["folder", "n_files"]).to_csv(
            AUDIT_DIR / "csv_folder_summary.csv", index=False)

    # Reporte
    print(f"[AUDIT] figures clasificadas: {len(fig_df)}")
    if not fig_df.empty:
        print(fig_df.groupby("status").size().to_string())
    print()
    print(f"[AUDIT] CSVs clasificados: {len(csv_df)}")
    if not csv_df.empty:
        print(csv_df.groupby("status").size().to_string())
    print(f"\nReportes en: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
