"""
regen_modified_figures.py — Regenera SOLO las figuras del layered pipeline
cuyo codigo de visualizacion fue modificado, a partir de los CSVs ya
existentes en outputs/metrics/layered_pipeline/. NO re-corre LOEO.

Figuras regeneradas:
    09_heatmap_model_vs_branch_MAE.png        (rotado: branches como filas)
    09_heatmap_model_vs_branch_R2.png         (rotado: branches como filas)
    09_sequential_comparison_dashboard_MAE.png   (mas alto, fonts mas grandes)
    09_sequential_comparison_dashboard_RMSE.png  (mas alto, fonts mas grandes)
    09_sequential_comparison_dashboard_R2.png    (mas alto, fonts mas grandes)
    09_model_evolution_MAE_LOEO.png           (titulo: 36 ramas)
    09_model_evolution_RMSE_LOEO.png
    09_model_evolution_R2_LOEO.png
    09_model_evolution_MAPE_LOEO.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from phm.layered_visuals import (
    plot_heatmap_model_vs_branch,
    plot_sequential_dashboard,
    plot_model_evolution,
)


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics" / "layered_pipeline"
FIG = ROOT / "outputs" / "figures" / "layered_pipeline"

assert METRICS.exists(), f"Falta {METRICS}"
FIG.mkdir(parents=True, exist_ok=True)


def _load(name: str) -> pd.DataFrame:
    p = METRICS / name
    if not p.exists():
        print(f"  [SKIP] {name} no encontrado")
        return pd.DataFrame()
    return pd.read_csv(p)


def main():
    all_metrics      = _load("09_all_metrics.csv")
    best_per_branch  = _load("09_branch_best_summary.csv")
    delta_df         = _load("09_delta_vs_baseline.csv")
    tuning_eff       = _load("09_tuning_effect_summary.csv")
    aug_eff          = _load("09_augmentation_effect_summary.csv")
    evolution        = _load("09_model_evolution_summary.csv")
    by_model         = _load("09_model_evolution_by_model.csv")

    print("\n=== HEATMAPS (rotados) ===")
    for metric, lib in [("MAE", True), ("R2", False)]:
        p = plot_heatmap_model_vs_branch(FIG, all_metrics, metric=metric,
                                          lower_is_better=lib)
        print(f"  {metric}: {p}")

    print("\n=== SEQUENTIAL DASHBOARDS ===")
    for metric in ("MAE", "RMSE", "R2"):
        p = plot_sequential_dashboard(FIG, best_per_branch, delta_df,
                                       tuning_eff, aug_eff, metric=metric)
        print(f"  {metric}: {p}")

    print("\n=== MODEL EVOLUTION ===")
    for metric, lib in [("MAE", True), ("RMSE", True),
                        ("R2", False), ("MAPE_%", True)]:
        # MAPE has a special filename suffix
        fname = "09_model_evolution_MAPE_LOEO" if metric == "MAPE_%" \
                else f"09_model_evolution_{metric}_LOEO"
        p = plot_model_evolution(FIG, evolution, by_model,
                                  metric=metric,
                                  lower_is_better=lib,
                                  filename=fname)
        print(f"  {metric}: {p}")

    print("\nLISTO.")


if __name__ == "__main__":
    main()
