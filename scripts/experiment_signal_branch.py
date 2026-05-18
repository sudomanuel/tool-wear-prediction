"""
Experimento aislado: bifurcacion por tipo de senal.

Compara 3 ramas en paralelo, todas con LOEO-CV y sin tuning:
    FUSION : todas las features (A + R + agregadas)         [203 features]
    SOLO_A : solo features que empiezan con A_              [~101 features]
    SOLO_R : solo features que empiezan con R_              [~99 features]

Para cada rama entrena todos los modelos baseline y genera:
    - Scatter VB_real vs VB_pred (45 deg) por modelo y rama
    - Barplot comparativo de MAE/R2 entre ramas
    - Tabla resumen

NO modifica nada del pipeline existente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

from phm.config import TARGET_COLUMN, EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL
from phm.modeling import all_baseline_models


# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "experiment_features.csv"
OUT_DIR = ROOT / "outputs" / "figures" / "signal_branch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_FEAT = {TARGET_COLUMN, EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL,
            "end_of_life", "is_augmented"}

BRANCH_COLORS = {
    "FUSION": "#5C3F8E",   # purpura
    "SOLO_A": "#B0324A",   # rojo (axial)
    "SOLO_R": "#1B7F5A",   # verde (rotacional)
}

MODEL_COLORS = {
    "DummyRegressor": "#9E9E9E", "Ridge": "#2196F3", "Lasso": "#03A9F4",
    "ElasticNet": "#00BCD4", "SVR": "#FF9800", "RandomForest": "#4CAF50",
    "XGBoost": "#8BC34A", "MLP": "#9C27B0",
}


# ----------------------------------------------------------------------------
# load
# ----------------------------------------------------------------------------
df = pd.read_csv(DATASET)
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols")

# eliminar columnas no numericas que no sean target/id
for col in list(df.columns):
    if col in NON_FEAT:
        continue
    if df[col].dtype == object:
        df = df.drop(columns=[col])

all_features = [c for c in df.columns if c not in NON_FEAT]
A_features   = [c for c in all_features if c.startswith("A_")]
R_features   = [c for c in all_features if c.startswith("R_")]
cross_feats  = [c for c in all_features if c not in A_features and c not in R_features]

print(f"  Axial (A_):  {len(A_features)}")
print(f"  Rot   (R_):  {len(R_features)}")
print(f"  Cross/agg:   {len(cross_feats)}  -> {cross_feats}")
print(f"  TOTAL:       {len(all_features)}")

BRANCHES = {
    "FUSION": all_features,
    "SOLO_A": A_features,
    "SOLO_R": R_features,
}

y      = df[TARGET_COLUMN].values
groups = df[EXPERIMENT_ID_COL].values
logo   = LeaveOneGroupOut()
n_folds = logo.get_n_splits(None, y, groups)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def compute_metrics(real, pred):
    real, pred = np.array(real), np.array(pred)
    mae  = float(np.mean(np.abs(real - pred)))
    rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
    ss_res = float(np.sum((real - pred) ** 2))
    ss_tot = float(np.sum((real - np.mean(real)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def run_loeo(X, y, groups, models):
    """Devuelve dict[model] = {'real':[], 'pred':[], 'exp':[]}."""
    res = {n: {"real": [], "pred": [], "exp": []} for n in models}
    for tr_idx, te_idx in logo.split(X, y, groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        eids = groups[te_idx]
        for name, pipe in models.items():
            fitted = clone(pipe)
            fitted.fit(X_tr, y_tr)
            y_pred = fitted.predict(X_te)
            res[name]["real"].extend(y_te.tolist())
            res[name]["pred"].extend(y_pred.tolist())
            res[name]["exp"].extend(eids.tolist())
    return res


def plot_scatter(ax, real, pred, color, title):
    real, pred = np.array(real), np.array(pred)
    mae, rmse, r2 = compute_metrics(real, pred)

    lo = min(real.min(), pred.min()) - 10
    hi = max(real.max(), pred.max()) + 10

    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5, label="Perfect")
    ax.scatter(real, pred, color=color, alpha=0.75, edgecolors="white",
               linewidths=0.5, s=55, zorder=3)
    try:
        m, b = np.polyfit(real, pred, 1)
        x_fit = np.linspace(lo, hi, 200)
        ax.plot(x_fit, m * x_fit + b, color=color, lw=1.5, alpha=0.85,
                label=f"Fit ({m:.2f}x)")
    except Exception:
        pass

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("VB_um real", fontsize=8)
    ax.set_ylabel("VB_um pred", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left")
    txt = f"MAE={mae:.1f}\nRMSE={rmse:.1f}\nR2={r2:.2f}"
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, fontsize=7,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, alpha=0.85))
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25, linestyle="--")


# ----------------------------------------------------------------------------
# run all branches
# ----------------------------------------------------------------------------
all_results = {}
all_metrics_rows = []

print(f"\nRunning LOEO-CV ({n_folds} folds) for {len(BRANCHES)} branches x "
      f"{len(all_baseline_models())} models...\n")

for branch_name, feats in BRANCHES.items():
    print(f"[{branch_name}] {len(feats)} features")
    X = df[feats].values.astype(float)
    models = all_baseline_models()
    res = run_loeo(X, y, groups, models)
    all_results[branch_name] = res
    for model_name, data in res.items():
        mae, rmse, r2 = compute_metrics(data["real"], data["pred"])
        all_metrics_rows.append({
            "branch": branch_name, "model": model_name,
            "n_features": len(feats),
            "MAE": mae, "RMSE": rmse, "R2": r2,
        })


# ----------------------------------------------------------------------------
# tabla resumen
# ----------------------------------------------------------------------------
df_metrics = pd.DataFrame(all_metrics_rows)
df_metrics_sorted = df_metrics.sort_values(["model", "branch"])

print("\n" + "=" * 80)
print("RESULTADOS POR MODELO Y RAMA (LOEO-CV)")
print("=" * 80)
print(f"{'Model':<16} {'Branch':<8} {'feats':>6} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print("-" * 80)
for _, r in df_metrics_sorted.iterrows():
    print(f"{r['model']:<16} {r['branch']:<8} {int(r['n_features']):>6} "
          f"{r['MAE']:>8.1f} {r['RMSE']:>8.1f} {r['R2']:>8.3f}")
print("=" * 80)

# guardar CSV
csv_path = OUT_DIR / "metrics_by_branch.csv"
df_metrics_sorted.to_csv(csv_path, index=False)
print(f"\n[OK] CSV: {csv_path}")


# ----------------------------------------------------------------------------
# figura 1: grid scatter (modelos x ramas)
# ----------------------------------------------------------------------------
models_list = list(all_baseline_models().keys())
branches_list = list(BRANCHES.keys())

n_models = len(models_list)
n_branches = len(branches_list)

fig, axes = plt.subplots(n_models, n_branches,
                         figsize=(n_branches * 3.8, n_models * 3.5))

for i, model_name in enumerate(models_list):
    for j, bname in enumerate(branches_list):
        ax = axes[i, j] if n_models > 1 else axes[j]
        data = all_results[bname][model_name]
        color = BRANCH_COLORS[bname]
        title = f"{model_name}  |  {bname}"
        plot_scatter(ax, data["real"], data["pred"], color, title)

fig.suptitle("Comparacion por rama de senal — LOEO-CV (sin tuning)",
             fontsize=15, fontweight="bold", y=1.0)
fig.tight_layout()
out1 = OUT_DIR / "scatter_grid_model_x_branch.png"
fig.savefig(out1, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out1.name}")


# ----------------------------------------------------------------------------
# figura 2: barras MAE por modelo, agrupado por rama
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# subplot 1: MAE
ax = axes[0]
x = np.arange(len(models_list))
width = 0.27
for i, bname in enumerate(branches_list):
    vals = [df_metrics[(df_metrics["model"] == m) &
                       (df_metrics["branch"] == bname)]["MAE"].iloc[0]
            for m in models_list]
    ax.bar(x + i * width - width, vals, width,
           label=bname, color=BRANCH_COLORS[bname], alpha=0.85,
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("MAE (um) — menor es mejor", fontsize=10)
ax.set_title("MAE por modelo y rama de senal", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.axhline(y=0, color="black", lw=0.8)

# subplot 2: R2
ax = axes[1]
for i, bname in enumerate(branches_list):
    vals = [df_metrics[(df_metrics["model"] == m) &
                       (df_metrics["branch"] == bname)]["R2"].iloc[0]
            for m in models_list]
    # clip valores muy negativos para visualizacion
    vals_plot = [max(v, -2) for v in vals]
    ax.bar(x + i * width - width, vals_plot, width,
           label=bname, color=BRANCH_COLORS[bname], alpha=0.85,
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("R2 — mayor es mejor (clip a -2)", fontsize=10)
ax.set_title("R2 por modelo y rama de senal", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.axhline(y=0, color="black", lw=0.8)
ax.set_ylim(-2.1, 1.05)

fig.suptitle("Comparacion FUSION vs SOLO_A vs SOLO_R",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
out2 = OUT_DIR / "barplot_mae_r2_by_branch.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out2.name}")


# ----------------------------------------------------------------------------
# figura 3: best model por rama (resumen ejecutivo)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, bname in enumerate(branches_list):
    sub = df_metrics[df_metrics["branch"] == bname].sort_values("MAE")
    best = sub.iloc[0]
    data = all_results[bname][best["model"]]
    ax = axes[i]
    color = BRANCH_COLORS[bname]
    title = f"{bname}  —  BEST: {best['model']}"
    plot_scatter(ax, data["real"], data["pred"], color, title)

fig.suptitle("Mejor modelo de cada rama de senal — LOEO-CV",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
out3 = OUT_DIR / "best_model_per_branch.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] {out3.name}")


# ----------------------------------------------------------------------------
# resumen final por rama
# ----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("RESUMEN: mejor modelo por rama")
print("=" * 80)
for bname in branches_list:
    sub = df_metrics[df_metrics["branch"] == bname].sort_values("MAE")
    best = sub.iloc[0]
    print(f"  {bname:<8} -> {best['model']:<16} "
          f"MAE={best['MAE']:.1f}  RMSE={best['RMSE']:.1f}  R2={best['R2']:.3f}  "
          f"({int(best['n_features'])} features)")
print("=" * 80)
print(f"\nFiguras: {OUT_DIR}")
