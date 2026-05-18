"""
Isolated baseline experiment: raw data, no tuning, LOEO-CV.
Generates one scatter plot (real vs predicted VB_um) per model.
Does NOT modify any existing pipeline outputs.
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
import matplotlib.gridspec as gridspec
from sklearn.model_selection import LeaveOneGroupOut

from phm.config import (
    TARGET_COLUMN,
    EXPERIMENT_ID_COL,
    TOOL_ID_COL,
    EXP_ORDER_COL,
)
from phm.modeling import all_baseline_models

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "experiment_features.csv"
OUT_DIR = ROOT / "outputs" / "figures" / "baseline_scatter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Experiments: {sorted(df[EXPERIMENT_ID_COL].unique())}")

NON_FEAT = {TARGET_COLUMN, EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL,
            "end_of_life", "is_augmented"}
feature_cols = [c for c in df.columns if c not in NON_FEAT]
print(f"Features: {len(feature_cols)}")

X = df[feature_cols].values
y = df[TARGET_COLUMN].values
groups = df[EXPERIMENT_ID_COL].values

# ── LOEO-CV ────────────────────────────────────────────────────────────────
logo = LeaveOneGroupOut()
models = all_baseline_models()

results = {name: {"real": [], "pred": [], "exp": []} for name in models}

n_folds = logo.get_n_splits(X, y, groups)
print(f"\nRunning LOEO-CV ({n_folds} folds) for {len(models)} models...\n")

for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    exp_ids = groups[test_idx]

    for name, pipe in models.items():
        from sklearn.base import clone
        fitted = clone(pipe)
        fitted.fit(X_train, y_train)
        y_pred = fitted.predict(X_test)

        results[name]["real"].extend(y_test.tolist())
        results[name]["pred"].extend(y_pred.tolist())
        results[name]["exp"].extend(exp_ids.tolist())

    print(f"  Fold {fold:2d}/{n_folds} — test exp {exp_ids[0]}")

print("\nAll folds done. Generating plots...\n")

# ── plot helpers ───────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Dummy":       "#9E9E9E",
    "Ridge":       "#2196F3",
    "Lasso":       "#03A9F4",
    "ElasticNet":  "#00BCD4",
    "SVR":         "#FF9800",
    "RandomForest":"#4CAF50",
    "XGBoost":     "#8BC34A",
    "MLP":         "#9C27B0",
}

def compute_metrics(real, pred):
    real, pred = np.array(real), np.array(pred)
    mae  = np.mean(np.abs(real - pred))
    rmse = np.sqrt(np.mean((real - pred) ** 2))
    ss_res = np.sum((real - pred) ** 2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def plot_scatter(name, real, pred, ax, color):
    real, pred = np.array(real), np.array(pred)
    mae, rmse, r2 = compute_metrics(real, pred)

    lo = min(real.min(), pred.min()) - 10
    hi = max(real.max(), pred.max()) + 10

    # 45° reference line
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="Perfect prediction")

    # scatter
    ax.scatter(real, pred, color=color, alpha=0.75, edgecolors="white",
               linewidths=0.5, s=60, zorder=3)

    # linear fit line
    m, b = np.polyfit(real, pred, 1)
    x_fit = np.linspace(lo, hi, 200)
    ax.plot(x_fit, m * x_fit + b, color=color, lw=1.8, linestyle="-",
            alpha=0.9, label=f"Fit (slope={m:.2f})")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("VB_um real (µm)", fontsize=9)
    ax.set_ylabel("VB_um predicho (µm)", fontsize=9)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")

    stats_txt = f"MAE={mae:.1f} µm\nRMSE={rmse:.1f} µm\nR²={r2:.3f}\nn={len(real)}"
    ax.text(0.97, 0.03, stats_txt, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.85))
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.25, linestyle="--")


# ── individual plots ────────────────────────────────────────────────────────
for name, data in results.items():
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    color = MODEL_COLORS.get(name, "#607D8B")
    plot_scatter(name, data["real"], data["pred"], ax, color)
    fig.suptitle(f"Baseline LOEO-CV — {name}", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / f"scatter_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── combined grid plot ──────────────────────────────────────────────────────
n_models = len(results)
ncols = 4
nrows = int(np.ceil(n_models / ncols))

fig = plt.figure(figsize=(ncols * 5, nrows * 5))
fig.suptitle("Baseline LOEO-CV — Todos los modelos\n(datos originales, sin tuning)",
             fontsize=14, fontweight="bold", y=1.01)

for i, (name, data) in enumerate(results.items()):
    ax = fig.add_subplot(nrows, ncols, i + 1)
    color = MODEL_COLORS.get(name, "#607D8B")
    plot_scatter(name, data["real"], data["pred"], ax, color)

# hide empty subplots
for j in range(i + 1, nrows * ncols):
    fig.add_subplot(nrows, ncols, j + 1).set_visible(False)

fig.tight_layout()
combined_path = OUT_DIR / "scatter_all_models.png"
fig.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Combined: {combined_path.name}")

# ── summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 55)
for name, data in results.items():
    mae, rmse, r2 = compute_metrics(data["real"], data["pred"])
    print(f"{name:<15} {mae:>8.1f} {rmse:>8.1f} {r2:>8.3f}")
print("=" * 55)
print(f"\nPlots saved to: {OUT_DIR}")
