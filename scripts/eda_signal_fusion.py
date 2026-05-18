"""
EDA — Análisis de la fusión de señales axial (A) y rotacional (R).
¿Tiene sentido fusionarlas? ¿Cuál contribuye más?
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
import seaborn as sns

from phm.config import TARGET_COLUMN, EXPERIMENT_ID_COL, NON_FEATURE_COLS

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "experiment_features.csv"
OUT_DIR = ROOT / "outputs" / "figures" / "eda_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATASET)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols")

NON_FEAT = {TARGET_COLUMN, EXPERIMENT_ID_COL, TOOL_ID_COL := "tool_id",
            EXP_ORDER_COL := "experiment_order", "end_of_life", "is_augmented"}
feature_cols = [c for c in df.columns if c not in NON_FEAT]

# Separar por dirección
A_features = [c for c in feature_cols if c.startswith("A_")]
R_features = [c for c in feature_cols if c.startswith("R_")]
both_features = set(A_features + R_features)
only_agg = [c for c in feature_cols if c not in both_features]

print(f"\nFeatures:")
print(f"  Axial (A_):      {len(A_features)}")
print(f"  Rotacional (R_): {len(R_features)}")
print(f"  Agregadas:       {len(only_agg)}")
print(f"  Total:           {len(feature_cols)}")

target = df[TARGET_COLUMN].values
X = df[feature_cols].fillna(df[feature_cols].mean()).values

# ── correlación con target ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRELACIÓN CON VB_um (target)")
print("=" * 70)

corr_all = []
for i, col in enumerate(feature_cols):
    c = np.corrcoef(X[:, i], target)[0, 1]
    corr_all.append((col, c))

corr_all.sort(key=lambda x: abs(x[1]), reverse=True)

A_corrs = [c for c in corr_all if c[0] in A_features]
R_corrs = [c for c in corr_all if c[0] in R_features]
agg_corrs = [c for c in corr_all if c[0] in only_agg]

print(f"\nTop 5 Axial (A):")
for feat, corr in A_corrs[:5]:
    print(f"  {feat:<40} r = {corr:+.3f}")

print(f"\nTop 5 Rotacional (R):")
for feat, corr in R_corrs[:5]:
    print(f"  {feat:<40} r = {corr:+.3f}")

print(f"\nTop 5 Agregadas:")
for feat, corr in agg_corrs[:5]:
    print(f"  {feat:<40} r = {corr:+.3f}")

# Stats por dirección
A_mean_abs_corr = np.mean([abs(c) for _, c in A_corrs])
R_mean_abs_corr = np.mean([abs(c) for _, c in R_corrs])
agg_mean_abs_corr = np.mean([abs(c) for _, c in agg_corrs]) if agg_corrs else 0

print(f"\nPromedio |correlación| por dirección:")
print(f"  Axial:       {A_mean_abs_corr:.3f}")
print(f"  Rotacional:  {R_mean_abs_corr:.3f}")
print(f"  Agregadas:   {agg_mean_abs_corr:.3f}")

# ── redundancia A vs R ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("REDUNDANCIA: Correlación entre direcciones (A vs R)")
print("=" * 70)

# Para cada contacto y feature, correlacionar A_pX_feat con R_pX_feat
redundancy = []
for contact in range(1, 7):
    for feat in ['rms', 'energy', 'mean', 'std']:
        A_col = f"A_p{contact}_{feat}"
        R_col = f"R_p{contact}_{feat}"
        if A_col in df.columns and R_col in df.columns:
            c = np.corrcoef(df[A_col].fillna(0), df[R_col].fillna(0))[0, 1]
            redundancy.append((f"p{contact}_{feat}", c))

redundancy.sort(key=lambda x: abs(x[1]), reverse=True)
print(f"\nCorrelación A_pX_feat vs R_pX_feat (misma posición):")
for feat, corr in redundancy[:10]:
    print(f"  {feat:<25} r = {corr:+.3f}")

mean_redundancy = np.mean([abs(c) for _, c in redundancy])
print(f"\nPromedio |correlación| A-R (mismo contacto): {mean_redundancy:.3f}")
if mean_redundancy > 0.7:
    print("  [ALTA REDUNDANCIA] — señales muy correlacionadas")
elif mean_redundancy > 0.4:
    print("  [REDUNDANCIA MODERADA] — algún solapamiento")
else:
    print("  [BAJA REDUNDANCIA] — señales complementarias")

# ── visualization 1: barplot top features por dirección ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Axial
A_top = A_corrs[:10]
ax = axes[0]
feats_a = [c.replace("A_", "") for c, _ in A_top]
corrs_a = [abs(c) for _, c in A_top]
colors_a = ["green" if c > 0 else "red" for _, c in A_top]
ax.barh(range(len(feats_a)), corrs_a, color=colors_a, alpha=0.7)
ax.set_yticks(range(len(feats_a)))
ax.set_yticklabels(feats_a, fontsize=8)
ax.set_xlabel("|Correlación| con VB_um", fontsize=9)
ax.set_title("Top 10 Axial (A)", fontweight="bold", fontsize=11)
ax.set_xlim(0, max(corrs_a) * 1.1)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

# Rotacional
R_top = R_corrs[:10]
ax = axes[1]
feats_r = [c.replace("R_", "") for c, _ in R_top]
corrs_r = [abs(c) for _, c in R_top]
colors_r = ["green" if c > 0 else "red" for _, c in R_top]
ax.barh(range(len(feats_r)), corrs_r, color=colors_r, alpha=0.7)
ax.set_yticks(range(len(feats_r)))
ax.set_yticklabels(feats_r, fontsize=8)
ax.set_xlabel("|Correlación| con VB_um", fontsize=9)
ax.set_title("Top 10 Rotacional (R)", fontweight="bold", fontsize=11)
ax.set_xlim(0, max(corrs_r) * 1.1)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

# Agregadas
ax = axes[2]
agg_top = agg_corrs[:10]
feats_agg = [c for c, _ in agg_top]
corrs_agg = [abs(c) for _, c in agg_top]
colors_agg = ["green" if c > 0 else "red" for _, c in agg_top]
ax.barh(range(len(feats_agg)), corrs_agg, color=colors_agg, alpha=0.7)
ax.set_yticks(range(len(feats_agg)))
ax.set_yticklabels(feats_agg, fontsize=8)
ax.set_xlabel("|Correlación| con VB_um", fontsize=9)
ax.set_title("Top 10 Agregadas", fontweight="bold", fontsize=11)
ax.set_xlim(0, max(corrs_agg) * 1.1)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

fig.suptitle("Contribución individual de cada dirección a VB_um",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "top_features_per_direction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[OK] Saved: top_features_per_direction.png")

# ── visualization 2: heatmap redundancia A vs R ──────────────────────────────
redundancy_matrix = np.zeros((6, 4))
feat_names = ['rms', 'energy', 'mean', 'std']

for contact in range(1, 7):
    for fi, feat in enumerate(feat_names):
        A_col = f"A_p{contact}_{feat}"
        R_col = f"R_p{contact}_{feat}"
        if A_col in df.columns and R_col in df.columns:
            c = np.corrcoef(df[A_col].fillna(0), df[R_col].fillna(0))[0, 1]
            redundancy_matrix[contact-1, fi] = abs(c)

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(redundancy_matrix, annot=True, fmt=".2f", cmap="RdYlGn_r",
            xticklabels=feat_names, yticklabels=[f"Contacto {i}" for i in range(1, 7)],
            cbar_kws={"label": "|Correlación| A vs R"}, ax=ax, vmin=0, vmax=1)
ax.set_title("Redundancia entre Axial y Rotacional por contacto",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "redundancy_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[OK] Saved: redundancy_heatmap.png")

# ── summary report ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RECOMENDACIÓN")
print("=" * 70)

ratio_A_to_R = A_mean_abs_corr / R_mean_abs_corr if R_mean_abs_corr > 0 else 1
print(f"\nAxial vs Rotacional:")
print(f"  A contribuye {A_mean_abs_corr:.3f} (promedio |r|)")
print(f"  R contribuye {R_mean_abs_corr:.3f} (promedio |r|)")
print(f"  Ratio A/R: {ratio_A_to_R:.2f}x")

if ratio_A_to_R > 1.5:
    print("\n  [NOTA] AXIAL domina mucho. Considerar:")
    print("    - Entrenar modelo SOLO con A_features")
    print("    - O ponderar features R más agresivamente")
elif ratio_A_to_R < 0.67:
    print("\n  [NOTA] ROTACIONAL domina. Considerar:")
    print("    - Entrenar modelo SOLO con R_features")
    print("    - O ponderar features A más agresivamente")
else:
    print("\n  [OK] Contribuyen por igual => fusión tiene sentido")

if mean_redundancy > 0.7:
    print(f"\n  [PROBLEMA] Alta redundancia A-R ({mean_redundancy:.3f})")
    print("    - Muchas features duplicadas")
    print("    - Prueba: seleccionar solo A O solo R")
elif mean_redundancy > 0.4:
    print(f"\n  [MODERADO] Redundancia media A-R ({mean_redundancy:.3f})")
    print("    - Hay solapamiento pero también info complementaria")
    print("    - Fusión está justificada")
else:
    print(f"\n  [OK] Baja redundancia A-R ({mean_redundancy:.3f})")
    print("    - Señales son complementarias")
    print("    - Fusión está bien justificada")

print("\n" + "=" * 70)
print(f"Figuras guardadas en: {OUT_DIR}")
