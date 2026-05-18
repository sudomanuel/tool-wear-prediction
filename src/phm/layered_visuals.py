"""
layered_visuals.py — figuras del pipeline por capas (LOEO-only).

Cada funcion responde una pregunta concreta y se nombra con el prefijo
numerico del paso (00_, 09_, ...) en el path final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

from .config import FIGURE_DPI, FIGURE_FORMAT
from .layered_pipeline import (
    FEATURE_SUBSETS, AUGMENTATION_STRATEGIES, parse_branch_id,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _build_branch_order() -> list:
    """36 ramas: por cada subset, 12 etapas (N x {ST, random, grid} +
    A x 3 strategies x 3 tuning)."""
    stages_per_subset = (
        ['N_ST', 'N_CT_random', 'N_CT_grid'] +
        [f'A_ST_{aug}' for aug in AUGMENTATION_STRATEGIES] +
        [f'A_CT_random_{aug}' for aug in AUGMENTATION_STRATEGIES] +
        [f'A_CT_grid_{aug}' for aug in AUGMENTATION_STRATEGIES]
    )
    return [f'{subset}_{stage}' for subset in FEATURE_SUBSETS for stage in stages_per_subset]


BRANCH_ORDER = _build_branch_order()

MODEL_ORDER = ['DummyRegressor', 'Ridge', 'Lasso', 'ElasticNet',
               'SVR', 'RandomForest', 'XGBoost', 'MLP']


def _save(fig, target_dir: Path, name: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def _branch_color(bid: str) -> str:
    """Color por (subset, data_branch). 6 colores: 3 subsets x {N, A}."""
    meta = parse_branch_id(bid)
    subset = meta.get('feature_subset', '')
    is_aug = meta.get('data_branch') == 'A'
    palette = {
        ('FUSION', False): '#7CA9C8',  # azul claro
        ('FUSION', True):  '#1F4E79',  # azul oscuro
        ('SOLO_A', False): '#E4AA88',  # naranja claro
        ('SOLO_A', True):  '#A0521E',  # naranja oscuro
        ('SOLO_R', False): '#8FCDB1',  # verde claro
        ('SOLO_R', True):  '#1B7F5A',  # verde oscuro
    }
    return palette.get((subset, is_aug), '#9E9E9E')


def _clip_high(values, lower_is_better=True, k=6.0):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3 or not lower_is_better:
        return None
    med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med))) or 1.0
    if vals.max() > med + k * mad:
        return float(med + (k - 2) * mad)
    return None


# =============================================================================
# 00 — Diagrama del flujo (LOEO-only)
# =============================================================================
def plot_layered_flow_diagram(target_dir: Path,
                              best_model_name: str = None,
                              best_mae: float = None,
                              filename: str = '00_layered_flow_diagram_no_holdout') -> Path:
    """
    Diagrama jerarquico del flujo — 4 niveles con trifurcacion inicial:
        D → {FUSION, SOLO_A, SOLO_R} → {N, A} → {ST, CT_r, CT_g} → LOEO → RANK → SHAP

    Layout:
      - Nivel 0: Dataset
      - Nivel 1: trifurcacion por subconjunto de features (FUSION / SOLO_A / SOLO_R)
      - Nivel 2: estado de aumentacion (N real / A aug) por cada subset — 6 nodos
      - Anotacion: x3 configs de tuning = 36 ramas totales
      - Nivel 3+: LOEO-CV → Ranking → SHAP
    """
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    COLOR = {
        'dataset': {'fc': '#FFF3CC', 'ec': '#A67C00', 'tc': '#5C4400'},
        'fusion':  {'fc': '#EDE0F5', 'ec': '#5C3F8E', 'tc': '#2D1F47'},
        'solo_a':  {'fc': '#FADCDC', 'ec': '#B0324A', 'tc': '#6B1F2D'},
        'solo_r':  {'fc': '#D4EDE5', 'ec': '#1B7F5A', 'tc': '#0F4A36'},
        'data_N':  {'fc': '#D1ECDF', 'ec': '#1B7F5A', 'tc': '#0F4A36'},
        'data_A':  {'fc': '#FBE5E5', 'ec': '#B0324A', 'tc': '#6B1F2D'},
        'cv':      {'fc': '#DCEAF7', 'ec': '#1F4E79', 'tc': '#102C44'},
        'final':   {'fc': '#E8E0F2', 'ec': '#5C3F8E', 'tc': '#2D1F47'},
        'shap':    {'fc': '#FFE6BD', 'ec': '#A86A00', 'tc': '#5C3A00'},
    }
    EDGE_GRAY = '#7A7A7A'

    def _box(cx, cy, w, h, text, palette, fontsize=10, weight='normal'):
        b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=0.2,rounding_size=0.6",
                           linewidth=1.6, edgecolor=palette['ec'],
                           facecolor=palette['fc'], zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize, color=palette['tc'],
                weight=weight, zorder=4)
        return dict(cx=cx, cy=cy, w=w, h=h,
                    top=(cx, cy + h / 2), bottom=(cx, cy - h / 2),
                    left=(cx - w / 2, cy), right=(cx + w / 2, cy))

    def _arrow(p_from, p_to, color=EDGE_GRAY, lw=1.4):
        a = FancyArrowPatch(p_from, p_to,
                            arrowstyle='->,head_length=8,head_width=6',
                            linewidth=lw, color=color,
                            shrinkA=0, shrinkB=2, zorder=2)
        ax.add_patch(a)

    def _line(p_from, p_to, color=EDGE_GRAY, lw=1.4):
        ax.plot([p_from[0], p_to[0]], [p_from[1], p_to[1]],
                color=color, linewidth=lw,
                solid_capstyle='round', zorder=1)

    # --- Nivel 0: Dataset ---
    D = _box(50, 93, 56, 5.5,
             'D — Dataset  (experiment_features.csv,  n = 10)',
             COLOR['dataset'], fontsize=11.5, weight='bold')

    # --- Nivel 1: trifurcacion por feature subset ---
    F  = _box(20, 82, 27, 7.5,
              'FUSION\n203 feat  (A + R + agg)',
              COLOR['fusion'], fontsize=9.5, weight='bold')
    SA = _box(50, 82, 27, 7.5,
              'SOLO_A\n~101 feat  (axial)',
              COLOR['solo_a'], fontsize=9.5, weight='bold')
    SR = _box(80, 82, 27, 7.5,
              'SOLO_R\n~99 feat  (rotac.)',
              COLOR['solo_r'], fontsize=9.5, weight='bold')

    # D → 3 subsets via mini-busbar
    d_bus_y = 89.25
    _line(D['bottom'], (D['cx'], d_bus_y), color='#A67C00', lw=1.5)
    _line((F['cx'], d_bus_y), (SR['cx'], d_bus_y), color='#A67C00', lw=1.5)
    _arrow((F['cx'],  d_bus_y), F['top'],  color=COLOR['fusion']['ec'], lw=1.7)
    _arrow((SA['cx'], d_bus_y), SA['top'], color=COLOR['solo_a']['ec'], lw=1.7)
    _arrow((SR['cx'], d_bus_y), SR['top'], color=COLOR['solo_r']['ec'], lw=1.7)

    # --- Nivel 2: N/A per subset (6 nodos) ---
    bw2, bh2 = 12.5, 5.5
    # FUSION subset
    fN  = _box(10,  69, bw2, bh2, 'F · N\n(real)', COLOR['data_N'], fontsize=8.5, weight='bold')
    fA  = _box(29,  69, bw2, bh2, 'F · A\n(aug)',  COLOR['data_A'], fontsize=8.5, weight='bold')
    # SOLO_A subset
    saN = _box(43,  69, 11,  bh2, 'SA · N\n(real)',COLOR['data_N'], fontsize=8.5, weight='bold')
    saA = _box(57,  69, 11,  bh2, 'SA · A\n(aug)', COLOR['data_A'], fontsize=8.5, weight='bold')
    # SOLO_R subset
    srN = _box(71,  69, bw2, bh2, 'SR · N\n(real)',COLOR['data_N'], fontsize=8.5, weight='bold')
    srA = _box(90,  69, 11.5,bh2, 'SR · A\n(aug)', COLOR['data_A'], fontsize=8.5, weight='bold')

    def _subset_busbar(parent, left_child, right_child, color, bus_y):
        _line(parent['bottom'], (parent['cx'], bus_y), color=color, lw=1.5)
        _line((left_child['cx'], bus_y), (right_child['cx'], bus_y), color=color, lw=1.5)
        _arrow((left_child['cx'],  bus_y), left_child['top'],  color=color, lw=1.3)
        _arrow((right_child['cx'], bus_y), right_child['top'], color=color, lw=1.3)

    _subset_busbar(F,  fN,  fA,  COLOR['fusion']['ec'],  75.5)
    _subset_busbar(SA, saN, saA, COLOR['solo_a']['ec'],   75.5)
    _subset_busbar(SR, srN, srA, COLOR['solo_r']['ec'],   75.5)

    # --- Anotacion lateral: x3 tuning = 36 ramas ---
    # Lines from all 6 N/A boxes → main busbar at y=62
    main_bus_y = 62.0
    all_na = (fN, fA, saN, saA, srN, srA)
    for node in all_na:
        _line(node['bottom'], (node['cx'], main_bus_y), color=EDGE_GRAY, lw=1.0)
    _line((fN['cx'], main_bus_y), (srA['cx'], main_bus_y), color=EDGE_GRAY, lw=1.8)

    ax.annotate(
        '× 3 configs de tuning por par:\n  ST   ·   CT_random   ·   CT_grid\n⟹   36 ramas en total',
        xy=(52, 59.5), xytext=(69, 56.5),
        ha='left', va='center', fontsize=9, color='#444444', style='italic',
        arrowprops=dict(arrowstyle='->', color='#999999', lw=0.9),
        bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFF0', ec='#BBBBBB', lw=0.9),
        zorder=5
    )

    # --- LOEO-CV ---
    LOEO = _box(50, 52, 60, 5.5,
                'LOEO-CV   ·   10 folds   ·   n_test = 1   ·   honesto',
                COLOR['cv'], fontsize=11.5, weight='bold')
    _arrow((50, main_bus_y), LOEO['top'], color=COLOR['cv']['ec'], lw=2.0)

    # --- Ranking ---
    RANK = _box(50, 39, 60, 5.5,
                'Ranking final   ·   MAE LOEO menor = mejor',
                COLOR['final'], fontsize=11.5, weight='bold')
    _arrow(LOEO['bottom'], RANK['top'], color=COLOR['final']['ec'], lw=1.7)

    # --- SHAP ---
    SHAP = _box(50, 26, 60, 5.5,
                'SHAP   ·   interpretabilidad sobre datos REALES',
                COLOR['shap'], fontsize=11.5, weight='bold')
    _arrow(RANK['bottom'], SHAP['top'], color=COLOR['shap']['ec'], lw=1.7)

    # --- Footer: mejor modelo ---
    if best_model_name is not None and best_mae is not None and np.isfinite(best_mae):
        ax.text(50, 13.5,
                f'Mejor LOEO:   {best_model_name}    ·    MAE = {best_mae:.2f} µm',
                ha='center', va='center', fontsize=12, color='#1F4E79',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.6', fc='#E8F1F8',
                          ec='#1F4E79', linewidth=1.4))

    ax.set_title(
        'Pipeline experimental por capas — LOEO-CV (sin hold-out)\n'
        '3 subsets × 2 estados de aug × 3 configs de tuning = 36 ramas',
        fontsize=13, pad=12, weight='bold', color='#1F2937')

    fig.tight_layout()
    return _save(fig, target_dir, filename)


# =============================================================================
# 09A — Branch performance (best por rama, una metrica)
# =============================================================================
def plot_branch_performance(target_dir: Path, best_per_branch: pd.DataFrame,
                            metric: str, lower_is_better: bool = True,
                            filename: str = None) -> Path:
    """Best `metric` por rama, con etiqueta = mejor modelo."""
    if best_per_branch.empty or metric not in best_per_branch.columns:
        return None
    sub = best_per_branch.copy()
    # Ordenar por rama estandar para que sea consistente entre figuras
    sub['__order'] = sub['branch_id'].apply(
        lambda b: BRANCH_ORDER.index(b) if b in BRANCH_ORDER else 99)
    sub = sub.sort_values('__order').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(sub) + 1)))
    colors = [_branch_color(b) for b in sub['branch_id']]
    # Resaltar el mejor global
    best_idx = sub[metric].idxmin() if lower_is_better else sub[metric].idxmax()
    colors[best_idx] = '#2E86AB'

    bars = ax.barh(sub['branch_id'][::-1], sub[metric][::-1],
                   color=colors[::-1], edgecolor='k', linewidth=0.4)
    ax.set_xlabel(metric)
    suffix = '— menor es mejor' if lower_is_better else '— mayor es mejor'
    ax.set_title(f'Best {metric} por rama (LOEO-CV)  {suffix}', fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)

    clip = _clip_high(sub[metric].values, lower_is_better)
    xmax = sub[metric].max() if clip is None else clip
    if clip is not None:
        ax.set_xlim(0, clip * 1.15)

    for i, (bar, v) in enumerate(zip(bars, sub[metric][::-1])):
        rev_i = len(sub) - 1 - i
        if not np.isfinite(v):
            continue
        label = sub['model'].iloc[rev_i]
        if clip is not None and v > clip:
            ax.text(clip * 1.05, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} ({label}) ⚠", va='center', ha='left',
                    fontsize=7, color='#D7263D')
        else:
            ax.text(v + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}  ({label})", va='center', ha='left', fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#2E86AB', edgecolor='k', label='best global'),
        Patch(facecolor='#7CA9C8', edgecolor='k', label='N — real'),
        Patch(facecolor='#D7906A', edgecolor='k', label='A — augmented'),
    ], loc='lower right', fontsize=8, frameon=True)
    fig.tight_layout()
    return _save(fig, target_dir, filename or f'09_branch_performance_{metric}')


# =============================================================================
# 09B — Heatmap modelo vs rama
# =============================================================================
def plot_heatmap_model_vs_branch(target_dir: Path, metrics_df: pd.DataFrame,
                                 metric: str = 'MAE',
                                 lower_is_better: bool = True,
                                 filename: str = None) -> Path:
    """Heatmap (modelo x rama) coloreado por `metric` LOEO."""
    df = metrics_df[metrics_df['validation_type'] == 'loeo'].copy()
    if df.empty or metric not in df.columns:
        return None
    # Pivot con ramas como filas (mejor para muchas ramas: 36) y modelos como cols (8)
    pivot = df.pivot_table(index='branch_id', columns='model',
                           values=metric, aggfunc='min')
    # Ordenar filas (branches) y columnas (models)
    row_order = [b for b in BRANCH_ORDER if b in pivot.index] + \
                [b for b in pivot.index if b not in BRANCH_ORDER]
    col_order = [m for m in MODEL_ORDER if m in pivot.columns] + \
                [m for m in pivot.columns if m not in MODEL_ORDER]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    # Clipping para escala visual (no afecta texto)
    vals_flat = pivot.values.flatten()
    vals_flat = vals_flat[np.isfinite(vals_flat)]
    vmax = float(np.percentile(vals_flat, 90)) if len(vals_flat) > 0 else None
    vmin = float(np.nanmin(pivot.values)) if np.isfinite(pivot.values).any() else None

    # Tamaño calculado: branches verticales (~36) caben mucho mejor en alto
    fig_h = max(8.0, 0.32 * len(row_order) + 2.5)
    fig_w = max(8.5, 0.95 * len(col_order) + 3.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = 'viridis_r' if lower_is_better else 'viridis'
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)

    # Anotar valores
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isfinite(v):
                ax.text(j, i, '·', ha='center', va='center',
                        fontsize=7, color='#888')
                continue
            color = 'white' if (vmax is not None and v > (vmin + vmax) / 2) else 'black'
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f'{metric}  ' +
                    ('(menor mejor)' if lower_is_better else '(mayor mejor)'))
    ax.set_title(f'Heatmap rama × modelo — {metric} LOEO-CV  ({len(row_order)} ramas × {len(col_order)} modelos)',
                 fontsize=12, pad=10)
    fig.tight_layout()
    return _save(fig, target_dir, filename or f'09_heatmap_model_vs_branch_{metric}')


# =============================================================================
# 09C — Delta vs baseline (best por rama)
# =============================================================================
def plot_delta_vs_baseline(target_dir: Path, delta_df: pd.DataFrame,
                           metric: str = 'MAE',
                           filename: str = None) -> Path:
    """Bar chart de delta_<metric>_vs_baseline. Negativo = mejora."""
    col = f'delta_{metric}_vs_baseline'
    if delta_df.empty or col not in delta_df.columns:
        return None
    sub = delta_df.copy()
    # Excluir baseline (delta=0 redundante? No, mostrarlo igual)
    sub['__order'] = sub['branch_id'].apply(
        lambda b: BRANCH_ORDER.index(b) if b in BRANCH_ORDER else 99)
    sub = sub.sort_values('__order').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(sub) + 1)))
    colors = ['#048A81' if v < 0 else ('#999999' if abs(v) < 1 else '#D7263D')
              for v in sub[col]]
    bars = ax.barh(sub['branch_id'][::-1], sub[col][::-1],
                    color=colors[::-1], edgecolor='k', linewidth=0.4)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'Δ {metric} (rama − baseline)   ·   negativo = mejora')
    baseline = sub['baseline_branch'].iloc[0] if 'baseline_branch' in sub.columns else 'N_ST'
    ax.set_title(f'Δ {metric} respecto al baseline {baseline}', fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)

    xabs = max(abs(sub[col].min()), abs(sub[col].max()), 1.0)
    for bar, v in zip(bars, sub[col][::-1]):
        if not np.isfinite(v):
            continue
        ax.text(v + (xabs * 0.02 if v >= 0 else -xabs * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.2f}",
                va='center', ha=('left' if v >= 0 else 'right'),
                fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#048A81', edgecolor='k', label='mejora (Δ < 0)'),
        Patch(facecolor='#999999', edgecolor='k', label='empate practico (|Δ|<1)'),
        Patch(facecolor='#D7263D', edgecolor='k', label='empeora (Δ > 0)'),
    ], loc='lower right', fontsize=8, frameon=True)
    fig.tight_layout()
    return _save(fig, target_dir,
                 filename or f'09_delta_{metric}_vs_baseline_N_ST')


# =============================================================================
# 09D — Tuning effect (ST vs Random vs Grid en cada data branch)
# =============================================================================
def plot_tuning_effect(target_dir: Path, tuning_effect_df: pd.DataFrame,
                       metric: str = 'MAE',
                       filename: str = None) -> Path:
    """Grouped bar: ST vs Random vs Grid por (data, aug)."""
    if tuning_effect_df.empty:
        return None
    sub = tuning_effect_df.copy()
    sub['group'] = sub.apply(
        lambda r: ('N' if r['data_branch'] == 'N' else 'A')
                  + (f"  ({r['augmentation_strategy']})" if r['augmentation_strategy'] != 'none' else ''),
        axis=1)
    # Orden estandar
    order = ['N',
             'A  (feature_noise)',
             'A  (feature_scaling)',
             'A  (grouped_scaling)']
    sub['__order'] = sub['group'].apply(lambda g: order.index(g) if g in order else 99)
    sub = sub.sort_values('__order').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(sub))
    w = 0.27
    bars_st = ax.bar(x - w, sub['best_MAE_ST'], width=w,
                      label='ST (sin tuning)', color='#7CA9C8',
                      edgecolor='k', linewidth=0.4)
    bars_rd = ax.bar(x,     sub['best_MAE_CT_random'], width=w,
                      label='CT_random', color='#F6BE7B',
                      edgecolor='k', linewidth=0.4)
    bars_gd = ax.bar(x + w, sub['best_MAE_CT_grid'], width=w,
                      label='CT_grid', color='#D7263D',
                      edgecolor='k', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(sub['group'], rotation=15, ha='right')
    ax.set_ylabel(f'Best {metric}')
    ax.set_title(f'Efecto del tuning — Best {metric} LOEO  ·  por (data, augmentation)',
                 fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    def _ann(bars):
        for b in bars:
            v = b.get_height()
            if not np.isfinite(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    _ann(bars_st); _ann(bars_rd); _ann(bars_gd)
    fig.tight_layout()
    return _save(fig, target_dir, filename or f'09_tuning_effect_{metric}')


# =============================================================================
# 09E — Random vs Grid
# =============================================================================
def plot_random_vs_grid(target_dir: Path, rg_df: pd.DataFrame,
                        metric: str = 'MAE',
                        filename: str = None) -> Path:
    """Comparacion directa Random vs Grid (paired bars)."""
    if rg_df.empty:
        return None
    sub = rg_df.copy()
    sub['group'] = sub.apply(
        lambda r: ('N' if r['data_branch'] == 'N' else 'A')
                  + (f"  ({r['augmentation_strategy']})" if r['augmentation_strategy'] != 'none' else ''),
        axis=1)
    order = ['N',
             'A  (feature_noise)',
             'A  (feature_scaling)',
             'A  (grouped_scaling)']
    sub['__order'] = sub['group'].apply(lambda g: order.index(g) if g in order else 99)
    sub = sub.sort_values('__order').reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(sub)); w = 0.35
    ax.bar(x - w/2, sub['best_MAE_random'], width=w,
           label='RandomizedSearchCV', color='#F6BE7B', edgecolor='k', linewidth=0.4)
    ax.bar(x + w/2, sub['best_MAE_grid'],   width=w,
           label='GridSearchCV', color='#D7263D', edgecolor='k', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(sub['group'], rotation=15, ha='right')
    ax.set_ylabel(f'Best {metric}')
    ax.set_title(f'Best {metric} LOEO — Random vs Grid', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()

    ax2 = axes[1]
    deltas = sub['delta_grid_minus_random'].values
    colors = ['#048A81' if d < -1 else ('#999999' if abs(d) <= 1 else '#D7263D') for d in deltas]
    bars = ax2.bar(x, deltas, color=colors, edgecolor='k', linewidth=0.4)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axhline(-1, color='#666', linewidth=0.5, linestyle='--')
    ax2.axhline(1,  color='#666', linewidth=0.5, linestyle='--')
    ax2.set_xticks(x); ax2.set_xticklabels(sub['group'], rotation=15, ha='right')
    ax2.set_ylabel(f'Δ {metric}  (grid − random)')
    ax2.set_title('Δ Grid − Random  ·  banda gris = empate practico (±1 µm)',
                  fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3)
    for b, v in zip(bars, deltas):
        if np.isfinite(v):
            ax2.text(b.get_x() + b.get_width() / 2, v,
                     f'{v:+.2f}', ha='center',
                     va=('bottom' if v >= 0 else 'top'), fontsize=8)

    fig.tight_layout()
    return _save(fig, target_dir, filename or f'09_random_vs_grid_{metric}')


# =============================================================================
# 09F — Augmentation effect
# =============================================================================
def plot_augmentation_effect(target_dir: Path, aug_df: pd.DataFrame,
                             metric: str = 'MAE',
                             filename: str = None) -> Path:
    """
    Grouped bar:
      eje x = tuning_method (none / random / grid)
      barras = N + 3 estrategias A
    """
    if aug_df.empty:
        return None

    strategies = ['none', 'feature_noise', 'feature_scaling', 'grouped_scaling']
    tunings = ['none', 'random', 'grid']

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(tunings))
    w = 0.2
    colors = {'none': '#7CA9C8', 'feature_noise': '#F6BE7B',
              'feature_scaling': '#D7906A', 'grouped_scaling': '#D7263D'}
    labels = {'none': 'N (sin aug)', 'feature_noise': 'A · feature_noise',
              'feature_scaling': 'A · feature_scaling',
              'grouped_scaling': 'A · grouped_scaling'}

    for i, strat in enumerate(strategies):
        sub = aug_df[aug_df['augmentation_strategy'] == strat]
        if sub.empty:
            continue
        ys = []
        for tm in tunings:
            row = sub[sub['tuning_method'] == tm]
            ys.append(float(row['best_MAE'].iloc[0]) if not row.empty else np.nan)
        offset = (i - (len(strategies) - 1) / 2) * w
        bars = ax.bar(x + offset, ys, width=w, label=labels[strat],
                       color=colors[strat], edgecolor='k', linewidth=0.4)
        for b, v in zip(bars, ys):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=6)

    ax.set_xticks(x); ax.set_xticklabels(['ST', 'CT_random', 'CT_grid'])
    ax.set_ylabel(f'Best {metric}')
    ax.set_title(f'Efecto de la augmentation — Best {metric} LOEO (por tuning method)',
                 fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return _save(fig, target_dir, filename or f'09_augmentation_effect_{metric}')


# =============================================================================
# 09G — Best model per branch
# =============================================================================
def plot_best_model_per_branch(target_dir: Path, best_per_branch: pd.DataFrame,
                               metric: str = 'MAE',
                               filename: str = None) -> Path:
    """Best modelo por rama — barra horizontal, etiqueta = modelo ganador."""
    if best_per_branch.empty:
        return None
    sub = best_per_branch.sort_values(metric).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.4 * len(sub) + 1)))
    colors = ['#2E86AB' if i == 0 else _branch_color(b)
              for i, b in enumerate(sub['branch_id'])]
    labels = (sub['branch_id'] + '  →  ' + sub['model']).tolist()
    bars = ax.barh(labels[::-1], sub[metric][::-1],
                    color=colors[::-1], edgecolor='k', linewidth=0.4)
    ax.set_xlabel(metric)
    ax.set_title(f'Mejor modelo por rama (LOEO, {metric} menor = mejor)',
                 fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    xmax = sub[metric].max()
    for bar, v in zip(bars, sub[metric][::-1]):
        if np.isfinite(v):
            ax.text(v + xmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va='center', ha='left', fontsize=8)
    fig.tight_layout()
    return _save(fig, target_dir, filename or f'09_best_model_per_branch_{metric}')


# =============================================================================
# 09H — Dashboard secuencial (2x2)
# =============================================================================
def plot_sequential_dashboard(target_dir: Path,
                              best_per_branch: pd.DataFrame,
                              delta_df: pd.DataFrame,
                              tuning_eff_df: pd.DataFrame,
                              aug_eff_df: pd.DataFrame,
                              metric: str = 'MAE',
                              filename: str = None) -> Path:
    """
    Dashboard 2x2:
      1. Best MAE por rama
      2. Δ MAE vs baseline (N_ST)
      3. Tuning effect (ST/Random/Grid)
      4. Augmentation effect (N + 3 strategies)
    """
    # Más alto para que las ~36 ramas en barh sean legibles
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # Panel 1: best metric por rama
    ax = axes[0, 0]
    if not best_per_branch.empty and metric in best_per_branch.columns:
        sub = best_per_branch.copy()
        sub['__order'] = sub['branch_id'].apply(
            lambda b: BRANCH_ORDER.index(b) if b in BRANCH_ORDER else 99)
        sub = sub.sort_values('__order').reset_index(drop=True)
        colors = [_branch_color(b) for b in sub['branch_id']]
        best_idx = sub[metric].idxmin()
        colors[best_idx] = '#2E86AB'
        ax.barh(sub['branch_id'][::-1], sub[metric][::-1],
                color=colors[::-1], edgecolor='k', linewidth=0.3)
        ax.set_title(f'1. Best {metric} por rama (LOEO)', fontsize=12)
        ax.set_xlabel(metric)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, axis='x', alpha=0.3)
        clip = _clip_high(sub[metric].values, True)
        if clip is not None:
            ax.set_xlim(0, clip * 1.15)
    else:
        ax.text(0.5, 0.5, '(sin filas)', ha='center', va='center')
        ax.axis('off')

    # Panel 2: delta vs baseline
    ax = axes[0, 1]
    col = f'delta_{metric}_vs_baseline'
    if not delta_df.empty and col in delta_df.columns:
        sub = delta_df.copy()
        sub['__order'] = sub['branch_id'].apply(
            lambda b: BRANCH_ORDER.index(b) if b in BRANCH_ORDER else 99)
        sub = sub.sort_values('__order').reset_index(drop=True)
        colors = ['#048A81' if v < -1 else ('#999999' if abs(v) <= 1 else '#D7263D')
                  for v in sub[col]]
        ax.barh(sub['branch_id'][::-1], sub[col][::-1],
                color=colors[::-1], edgecolor='k', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=0.8)
        baseline_branch = sub['baseline_branch'].iloc[0] if 'baseline_branch' in sub.columns else 'N_ST'
        ax.set_title(f'2. Δ {metric} vs baseline {baseline_branch}', fontsize=12)
        ax.set_xlabel(f'Δ {metric}')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, '(sin filas)', ha='center', va='center')
        ax.axis('off')

    # Panel 3: tuning effect
    ax = axes[1, 0]
    if not tuning_eff_df.empty:
        sub = tuning_eff_df.copy()
        sub['group'] = sub.apply(
            lambda r: ('N' if r['data_branch'] == 'N' else 'A')
                      + (f"  ({r['augmentation_strategy']})" if r['augmentation_strategy'] != 'none' else ''),
            axis=1)
        order = ['N', 'A  (feature_noise)', 'A  (feature_scaling)', 'A  (grouped_scaling)']
        sub['__order'] = sub['group'].apply(lambda g: order.index(g) if g in order else 99)
        sub = sub.sort_values('__order').reset_index(drop=True)
        x = np.arange(len(sub)); w = 0.27
        ax.bar(x - w, sub['best_MAE_ST'], w, label='ST', color='#7CA9C8', edgecolor='k', linewidth=0.3)
        ax.bar(x,     sub['best_MAE_CT_random'], w, label='Random', color='#F6BE7B', edgecolor='k', linewidth=0.3)
        ax.bar(x + w, sub['best_MAE_CT_grid'],   w, label='Grid', color='#D7263D', edgecolor='k', linewidth=0.3)
        ax.set_xticks(x); ax.set_xticklabels(sub['group'], rotation=15, ha='right', fontsize=9)
        ax.set_title(f'3. Tuning effect — Best {metric}', fontsize=11)
        ax.set_ylabel(metric); ax.grid(True, axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, '(sin filas)', ha='center', va='center')
        ax.axis('off')

    # Panel 4: augmentation effect (solo ST para no saturar)
    ax = axes[1, 1]
    if not aug_eff_df.empty:
        sub = aug_eff_df[aug_eff_df['tuning_method'] == 'none'].copy()
        strat_order = ['none', 'feature_noise', 'feature_scaling', 'grouped_scaling']
        sub['__order'] = sub['augmentation_strategy'].apply(
            lambda s: strat_order.index(s) if s in strat_order else 99)
        sub = sub.sort_values('__order').reset_index(drop=True)
        colors = ['#7CA9C8'] + ['#D7906A'] * (len(sub) - 1)
        ax.bar(sub['augmentation_strategy'], sub['best_MAE'],
               color=colors, edgecolor='k', linewidth=0.3)
        ax.set_title(f'4. Augmentation effect (ST, sin tuning) — Best {metric}',
                     fontsize=11)
        ax.set_ylabel(metric)
        ax.grid(True, axis='y', alpha=0.3)
        for i, v in enumerate(sub['best_MAE']):
            if np.isfinite(v):
                ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    else:
        ax.text(0.5, 0.5, '(sin filas)', ha='center', va='center')
        ax.axis('off')

    fig.suptitle(f'Dashboard secuencial — {metric}  (LOEO-CV, sin hold-out)',
                 fontsize=13, y=1.00)
    fig.tight_layout()
    return _save(fig, target_dir, filename or f'09_sequential_comparison_dashboard_{metric}')


# =============================================================================
# 09I — Evolucion del modelo a traves de las 12 ramas del pipeline
# =============================================================================
def plot_model_evolution(target_dir: Path,
                         evolution_df: pd.DataFrame,
                         by_model_df: pd.DataFrame,
                         metric: str = 'MAE',
                         lower_is_better: bool = True,
                         filename: str = None) -> Path:
    """
    Una sola figura que cuenta TODO el analisis rama por rama (12 ramas):

      - Eje X: 12 ramas en orden de progresion del pipeline
        (N_ST → N_CT_random → N_CT_grid → A_ST_{3} → A_CT_random_{3}
         → A_CT_grid_{3}).
      - Eje Y: `metric` (MAE por defecto).
      - Linea principal (azul oscura, gruesa, diamantes) = mejor modelo
        en cada rama. Etiqueta arriba indica que modelo gano.
      - Lineas finas (top-3 modelos) = el mismo modelo recorre las 12
        ramas. Asi se ve si un modelo (ej. ElasticNet) domina toda la
        progresion o si el ganador cambia segun la rama.
      - Bandas de color de fondo por "familia" de ramas (N, N·tuning,
        A·ST, A·random, A·grid).
      - Linea horizontal punteada = baseline N_ST.
      - Anotacion debajo de cada punto = Δ MAE vs rama PREVIA (verde si
        baja, rojo si sube, gris si <0.5 µm — empate practico).
      - Marca dorada en el "BEST OVERALL".

    Si la linea principal queda plana → la complejidad no aporta.
    Si sube → la complejidad esta empeorando el desempeno.
    """
    if evolution_df.empty or metric not in evolution_df.columns:
        return None
    df = evolution_df.copy().sort_values('stage_order').reset_index(drop=True)

    from .layered_pipeline import FAMILY_COLORS

    fig, ax = plt.subplots(figsize=(16, 7.5))

    # --- Bandas de color por familia ---
    families_seen = []
    band_starts = {}
    for _, r in df.iterrows():
        fam = r.get('family', '')
        if fam not in band_starts:
            band_starts[fam] = r['stage_order']
        families_seen.append((r['stage_order'], fam))
    # Construir bandas continuas
    if families_seen:
        # encontrar rangos por familia
        ranges = []
        cur_fam = families_seen[0][1]
        cur_start = families_seen[0][0]
        for i in range(1, len(families_seen)):
            x, f = families_seen[i]
            if f != cur_fam:
                ranges.append((cur_fam, cur_start, families_seen[i-1][0]))
                cur_fam = f; cur_start = x
        ranges.append((cur_fam, cur_start, families_seen[-1][0]))
        for fam, x0, x1 in ranges:
            ax.axvspan(x0 - 0.45, x1 + 0.45,
                        color=FAMILY_COLORS.get(fam, '#EEEEEE'),
                        alpha=0.08, zorder=0)
            # Texto de la familia centrado arriba
            ax.text((x0 + x1) / 2, 1.02, fam,
                     transform=ax.get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=9,
                     color=FAMILY_COLORS.get(fam, '#333'), weight='bold')

    # --- Lineas finas: top-N modelos a traves de las 12 ramas ---
    if not by_model_df.empty and metric in by_model_df.columns:
        models = list(by_model_df['model'].unique())
        palette = ['#7CA9C8', '#F2B57C', '#A0A05D', '#8E6FB6', '#6FB67C']
        for i, m in enumerate(models):
            sub = by_model_df[by_model_df['model'] == m].sort_values('stage_order')
            color = palette[i % len(palette)]
            ax.plot(sub['stage_order'], sub[metric],
                    marker='o', markersize=5, linewidth=1.3,
                    color=color, alpha=0.75, label=m, zorder=2)

    # --- Linea principal: best per branch ---
    ax.plot(df['stage_order'], df[metric],
            marker='D', markersize=11, linewidth=2.8,
            color='#1F4E79', label='Best per branch (winner)',
            markeredgecolor='black', markeredgewidth=0.7, zorder=5)

    # Baseline horizontal
    baseline_val = df[df['stage_order'] == 1][metric]
    if not baseline_val.empty and np.isfinite(baseline_val.iloc[0]):
        bv = float(baseline_val.iloc[0])
        ax.axhline(bv, color='#1F4E79', linewidth=0.9,
                   linestyle='--', alpha=0.6,
                   label=f'Baseline N_ST = {bv:.2f}')

    # --- Estrella en BEST OVERALL ---
    if df['MAE'].notna().any():
        idx_best = df[metric].idxmin() if lower_is_better else df[metric].idxmax()
        rb = df.loc[idx_best]
        ax.plot(rb['stage_order'], rb[metric], marker='*',
                markersize=25, color='#FFD700',
                markeredgecolor='black', markeredgewidth=1.0,
                label='BEST OVERALL', zorder=7)

    # --- Anotaciones por rama: modelo arriba, delta abajo ---
    for i, r in df.iterrows():
        v = r[metric]
        if not np.isfinite(v):
            continue
        mdl = r.get('model', '')
        # Anotacion del modelo arriba
        ax.annotate(mdl,
                    xy=(r['stage_order'], v),
                    xytext=(0, 14), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5,
                    color='#1F4E79', weight='bold')
        # Delta MAE debajo (solo para metrica MAE)
        if metric == 'MAE' and i > 0:
            d = r.get('delta_MAE_vs_previous_stage', float('nan'))
            if np.isfinite(d):
                color = '#048A81' if d < -0.5 else ('#D7263D' if d > 0.5 else '#777')
                sign = '+' if d >= 0 else ''
                ax.annotate(f"Δ{sign}{d:.2f}",
                            xy=(r['stage_order'], v),
                            xytext=(0, -22), textcoords='offset points',
                            ha='center', va='top', fontsize=7,
                            color=color, weight='bold')

    ax.set_xticks(df['stage_order'])
    ax.set_xticklabels(df['short_label'], rotation=35, ha='right', fontsize=8)
    direction = '(menor = mejor)' if lower_is_better else '(mayor = mejor)'
    ax.set_ylabel(f'{metric}  {direction}', fontsize=10)
    n_ramas = len(df)
    ax.set_title(
        f'Evolucion del modelo rama por rama  —  {metric} (LOEO-CV, las {n_ramas} ramas)\n'
        f'Bandas: familia de configuracion  ·  diamante azul: ganador de la rama  ·  estrella dorada: best overall',
        fontsize=11, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=8, frameon=True, ncol=2)

    ax.margins(x=0.03)
    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.16
    ax.set_ylim(ymin - pad, ymax + pad)

    fig.tight_layout()
    return _save(fig, target_dir,
                  filename or f'09_model_evolution_{metric}_LOEO')


# =============================================================================
# 09J — Real vs Prediccion multi-overlay (puntitos de colores)
# =============================================================================
def plot_actual_vs_predicted_multi(target_dir: Path,
                                    selections: list,
                                    filename: str = None) -> Path:
    """
    Scatter de Real (VB_um) vs Prediccion, con multiples configuraciones
    superpuestas como series de puntos de colores distintos:

      1. Baseline (N_ST)              · azul oscuro
      2. Mejor tuneado N_CT           · azul claro
      3. Mejor A_ST                   · naranja
      4. Mejor A_CT (aug + tuning)    · marron
      5. BEST GLOBAL                  · rojo

    Cada serie tiene 10 puntos (1 por experimento). Linea negra y=x.
    Lineas de error gris claro conectan real ↔ prediccion para el best
    global, para resaltar magnitud del error.
    """
    if not selections:
        return None

    # Rango comun
    all_real = np.concatenate([s['y_real'] for s in selections])
    all_pred = np.concatenate([s['y_pred'] for s in selections])
    mn = float(np.nanmin(np.concatenate([all_real, all_pred]))) - 5
    mx = float(np.nanmax(np.concatenate([all_real, all_pred]))) + 5

    fig, ax = plt.subplots(figsize=(11, 9))

    # Diagonal y=x
    ax.plot([mn, mx], [mn, mx], '--', color='black', linewidth=1.2,
            alpha=0.6, label='y = x (prediccion perfecta)')

    # Banda ±20 µm (referencia visual)
    xs = np.linspace(mn, mx, 100)
    ax.fill_between(xs, xs - 20, xs + 20, color='#CCCCCC', alpha=0.20,
                    label='±20 µm', zorder=0)

    # Series
    markers = ['o', 's', '^', 'D', '*']
    for i, s in enumerate(selections):
        is_best = 'BEST GLOBAL' in s['label']
        marker = '*' if is_best else markers[i % len(markers)]
        msize  = 320 if is_best else 110
        edge   = 1.3 if is_best else 0.6
        ax.scatter(s['y_real'], s['y_pred'],
                    s=msize, marker=marker, c=s['color'],
                    edgecolor='black', linewidth=edge,
                    alpha=0.85 if not is_best else 0.95,
                    label=f"{s['label']}\n"
                          f"   {s['model']} · {s['branch_id']} · MAE={s['mae']:.2f}",
                    zorder=6 if is_best else 4)
        # Lineas verticales de error solo para el best global
        if is_best:
            for xr, yp in zip(s['y_real'], s['y_pred']):
                ax.plot([xr, xr], [xr, yp], color=s['color'], alpha=0.4,
                        linewidth=1.0, zorder=3)

    # Anotar experiment_id sobre los puntos del best global
    best_sel = next((s for s in selections if 'BEST GLOBAL' in s['label']), None)
    if best_sel is not None:
        for eid, xr, yp in zip(best_sel['eids'], best_sel['y_real'], best_sel['y_pred']):
            ax.annotate(f'exp{int(eid)}',
                        xy=(xr, yp), xytext=(7, 7),
                        textcoords='offset points', fontsize=7,
                        color='#666')

    ax.set_xlabel('VB_um real  (µm)', fontsize=11)
    ax.set_ylabel('VB_um predicho  (µm)', fontsize=11)
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(
        'Real vs Prediccion  —  configuraciones del pipeline superpuestas (LOEO-CV)\n'
        'Cada color = una rama: baseline, tuneado, augmented, augmented+tuneado, best global',
        fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, frameon=True,
               bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return _save(fig, target_dir,
                  filename or '09_actual_vs_predicted_multi_LOEO')


# =============================================================================
# 09K — Residuals multi-overlay (mismo concepto pero residuos por experimento)
# =============================================================================
def plot_residuals_by_experiment_multi(target_dir: Path,
                                        selections: list,
                                        filename: str = None) -> Path:
    """
    Para cada experimento (eje X), grafica el residuo (pred - real) de
    cada configuracion seleccionada como puntos de colores. Permite ver
    si todos los modelos fallan en los mismos experimentos.
    """
    if not selections:
        return None

    # Reunir todos los experiment_ids unicos
    all_eids = sorted(set(int(e) for s in selections for e in s['eids']))
    x_map = {e: i for i, e in enumerate(all_eids)}

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axhline(0, color='black', linewidth=1.0, alpha=0.7)
    ax.fill_between(range(len(all_eids)), -10, 10, color='#CCCCCC', alpha=0.2,
                     label='±10 µm')

    markers = ['o', 's', '^', 'D', '*']
    width = 0.15
    for i, s in enumerate(selections):
        is_best = 'BEST GLOBAL' in s['label']
        marker = '*' if is_best else markers[i % len(markers)]
        offset = (i - (len(selections) - 1) / 2) * width
        for eid, yr, yp in zip(s['eids'], s['y_real'], s['y_pred']):
            xpos = x_map[int(eid)] + offset
            res = float(yp - yr)
            ax.scatter(xpos, res, s=(280 if is_best else 90),
                        marker=marker, c=s['color'],
                        edgecolor='black', linewidth=(1.2 if is_best else 0.5),
                        alpha=(0.95 if is_best else 0.85),
                        zorder=(6 if is_best else 4),
                        label=(f"{s['label']}  ({s['model']} · {s['branch_id']})"
                               if eid == s['eids'][0] else None))
    ax.set_xticks(range(len(all_eids)))
    ax.set_xticklabels([f'exp{e}' for e in all_eids])
    ax.set_xlabel('Experimento'); ax.set_ylabel('Residuo  (pred − real)  µm')
    ax.set_title(
        'Residuos por experimento — configuraciones del pipeline superpuestas (LOEO-CV)\n'
        'Banda gris ±10 µm = error tolerable.  Mismo experimento → multiples puntos = varias configs',
        fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    # Deduplicar leyenda
    h, l = ax.get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for hi, li in zip(h, l):
        if li not in seen and li:
            seen.add(li); h2.append(hi); l2.append(li)
    ax.legend(h2, l2, loc='upper left', fontsize=8, frameon=True,
               bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return _save(fig, target_dir,
                  filename or '09_residuals_by_experiment_multi_LOEO')
