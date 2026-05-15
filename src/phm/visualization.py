"""
visualization.py — plots por etapa.

Cada funcion `_in(dir, ...)` acepta un directorio destino para
permitir subcarpetas por etapa (holdout/, loeo/, tuning/, augmentation/).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from .config import FIGURES_DIR, FIGURE_DPI, FIGURE_FORMAT


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _save(fig, target_dir: Path, name: str):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.{FIGURE_FORMAT}"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Generic
# -----------------------------------------------------------------------------
def bar_metric_in(target_dir, df: pd.DataFrame, metric: str, title: str,
                  lower_is_better: bool = True, filename: str = None,
                  horizontal: bool = None,
                  clip_outliers: bool = True) -> Path:
    """
    Bar chart de una metrica.
      - horizontal=None -> elige automaticamente: horizontal si >10 barras o
        si algun label tiene > 15 caracteres.
      - clip_outliers=True -> si un valor cae > 3 sigma fuera de la mediana,
        clipea la escala y anota el valor real fuera de la barra.
    """
    df = df.sort_values(metric, ascending=lower_is_better).reset_index(drop=True)
    n = len(df)
    labels = df['model'].astype(str).tolist()
    values = df[metric].astype(float).values

    if horizontal is None:
        max_label_len = max((len(s) for s in labels), default=0)
        horizontal = (n > 10) or (max_label_len > 15)

    # outlier clipping (solo para escala visual; el numero real se anota)
    clip_low, clip_high = None, None
    if clip_outliers and n >= 3 and np.isfinite(values).any():
        vals = values[np.isfinite(values)]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) or 1.0
        thr = 6.0 * mad   # mas generoso que 3 sigma para no clipear normales
        out_mask = np.abs(values - med) > thr
        if out_mask.any():
            keep = values[~out_mask & np.isfinite(values)]
            if len(keep) > 0:
                lo = float(min(keep.min(), 0)) - 0.1 * (keep.max() - keep.min() + 1)
                hi = float(keep.max()) + 0.2 * (keep.max() - keep.min() + 1)
                clip_low, clip_high = lo, hi

    if horizontal:
        height = max(3.5, 0.35 * n + 1.2)
        fig, ax = plt.subplots(figsize=(9.5, height))
        # mejor abajo -> arriba en el plot
        order = range(n - 1, -1, -1) if not lower_is_better else range(n)
        ylab = [labels[i] for i in order]
        yval = [values[i] for i in order]
        colors = ['#2E86AB' if i == (0 if lower_is_better else n - 1) else '#A0A0A0'
                  for i in order]
        bars = ax.barh(ylab, yval, color=colors, edgecolor='k', linewidth=0.5)
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
        if clip_low is not None:
            ax.set_xlim(clip_low, clip_high)
        # anotacion de valor al final de cada barra (incluyendo outliers clipeados)
        for bar, v in zip(bars, yval):
            if not np.isfinite(v):
                continue
            xpos = bar.get_width()
            clipped = (clip_high is not None and v > clip_high) or \
                      (clip_low  is not None and v < clip_low)
            if clipped:
                xpos = clip_high if v > 0 else clip_low
                text = f"{v:.2f} ⚠"
            else:
                text = f"{v:.2f}"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f" {text}", va='center', ha='left' if v >= 0 else 'right',
                    fontsize=8)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * n + 2), 5))
        colors = ['#2E86AB' if i == 0 else '#A0A0A0' for i in range(n)]
        bars = ax.bar(labels, values, color=colors, edgecolor='k', linewidth=0.5)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment('right')
        ax.grid(True, axis='y', alpha=0.3)
        if clip_low is not None:
            ax.set_ylim(clip_low, clip_high)
        for bar, v in zip(bars, values):
            if not np.isfinite(v):
                continue
            clipped = (clip_high is not None and v > clip_high) or \
                      (clip_low  is not None and v < clip_low)
            if clipped:
                ypos = clip_high * 0.97 if v > 0 else clip_low * 0.97
                text = f"{v:.2f} ⚠"
            else:
                ypos = v
                text = f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, text,
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
        fig.tight_layout()
    return _save(fig, target_dir, filename or f"{metric.lower()}_comparison")


def grouped_bar_metric_in(target_dir, df: pd.DataFrame,
                          group_col: str, sub_col: str,
                          metric: str, title: str,
                          lower_is_better: bool = True,
                          filename: str = None) -> Path:
    """
    Barras agrupadas: una barra por (group, sub). Util cuando hay muchas
    combinaciones (ej: 8 modelos × 4 estrategias) que como barras planas
    se vuelven ilegibles.
    """
    pivot = df.pivot_table(index=group_col, columns=sub_col, values=metric)
    # ordenar grupos por el mejor (menor o mayor segun lower_is_better)
    if lower_is_better:
        pivot = pivot.reindex(pivot.min(axis=1).sort_values().index)
    else:
        pivot = pivot.reindex(pivot.max(axis=1).sort_values(ascending=False).index)

    groups = pivot.index.tolist()
    subs   = pivot.columns.tolist()
    n_g, n_s = len(groups), len(subs)
    width = 0.8 / max(n_s, 1)
    x = np.arange(n_g)

    # paleta consistente para subs
    base_colors = ['#2E86AB', '#F18F01', '#048A81', '#D7263D',
                   '#7B2D26', '#5D5C61', '#379392', '#A23B72']
    color_map = {s: base_colors[i % len(base_colors)] for i, s in enumerate(subs)}

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * n_g + 2), 5.5))
    for i, s in enumerate(subs):
        vals = pivot[s].values
        offset = (i - (n_s - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=str(s),
               color=color_map[s], edgecolor='k', linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.axhline(0, color='k', lw=0.6)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(title=sub_col, fontsize=9, frameon=True, loc='best')

    # clipping para outliers tipo MLP+feature_noise R²=-1.20
    vals_all = pivot.values.flatten()
    vals_all = vals_all[np.isfinite(vals_all)]
    if len(vals_all) >= 3:
        med = np.median(vals_all)
        mad = np.median(np.abs(vals_all - med)) or 1.0
        thr = 6.0 * mad
        normales = vals_all[np.abs(vals_all - med) <= thr]
        if len(normales) >= 3 and len(normales) < len(vals_all):
            lo = float(min(normales.min(), 0)) - 0.1 * (normales.max() - normales.min() + 1)
            hi = float(normales.max()) + 0.2 * (normales.max() - normales.min() + 1)
            ax.set_ylim(lo, hi)
            # anotar outliers en el borde
            for j, s in enumerate(subs):
                for i, g in enumerate(groups):
                    v = pivot.loc[g, s]
                    if not np.isfinite(v): continue
                    if v > hi or v < lo:
                        offset = (j - (n_s - 1) / 2) * width
                        ypos = hi * 0.97 if v > 0 else lo * 0.97
                        ax.text(i + offset, ypos, f"{v:.1f}⚠",
                                ha='center', va='top' if v < 0 else 'bottom',
                                fontsize=7, color='#D7263D')

    fig.tight_layout()
    return _save(fig, target_dir, filename or f"{metric.lower()}_grouped")


def actual_vs_predicted_in(target_dir, y_dict: dict, title: str,
                           filename: str = "actual_vs_predicted_best") -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    all_vals = []
    for name, (y_t, y_p) in y_dict.items():
        ax.scatter(y_t, y_p, label=name, alpha=0.85, s=70,
                   edgecolors='k', linewidths=0.6)
        all_vals.extend(list(y_t) + list(y_p))
    if all_vals:
        lo, hi = min(all_vals) - 10, max(all_vals) + 10
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2, label='prediccion perfecta')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('VB_um real')
    ax.set_ylabel('VB_um predicho')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, target_dir, filename)


def residuals_plot_in(target_dir, y_dict: dict, title: str,
                      filename: str = "residuals_best") -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (y_t, y_p) in y_dict.items():
        res = np.asarray(y_p) - np.asarray(y_t)
        ax.scatter(y_t, res, label=name, alpha=0.85, s=70,
                   edgecolors='k', linewidths=0.6)
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('VB_um real')
    ax.set_ylabel('residual (predicho - real)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, target_dir, filename)


def residuals_by_experiment_in(target_dir, pred_df: pd.DataFrame,
                               filename: str = "residuals_by_experiment") -> Path:
    """Bar chart de residual por experimento, separado por modelo."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = sorted(pred_df['model'].unique())
    eids   = sorted(pred_df['experiment_id'].unique())
    width  = 0.8 / max(len(models), 1)
    x_base = np.arange(len(eids))
    for i, m in enumerate(models):
        sub = pred_df[pred_df['model'] == m].set_index('experiment_id').reindex(eids)
        offsets = x_base + (i - len(models) / 2 + 0.5) * width
        ax.bar(offsets, sub['residual'].values, width=width, label=m,
               edgecolor='k', linewidth=0.4)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels(eids)
    ax.set_xlabel('experiment_id')
    ax.set_ylabel('residual (predicho - real)')
    ax.set_title('LOEO — Residuals por experimento')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    return _save(fig, target_dir, filename)


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------
def plot_feature_missingness(target_dir, df: pd.DataFrame,
                             feature_cols: list,
                             filename: str = "feature_missingness") -> Path:
    miss = df[feature_cols].isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(miss) + 1)))
    if miss.empty:
        ax.text(0.5, 0.5, 'Sin valores faltantes',
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
    else:
        ax.barh(miss.index[::-1], miss.values[::-1],
                color='#D7263D', edgecolor='k', linewidth=0.4)
        ax.set_xlabel('# valores faltantes')
        ax.set_title('Missingness por feature')
        ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return _save(fig, target_dir, filename)


def plot_correlation_heatmap_top(target_dir, df: pd.DataFrame,
                                 feature_cols: list, target_col: str,
                                 top_n: int = 30,
                                 filename: str = "feature_correlation_heatmap_top30") -> Path:
    """
    Heatmap con las top_n features mas correlacionadas con el target.
    Resalta la fila/columna del target para que se lea de un vistazo
    que features ayudan/perjudican.
    """
    sub = df[feature_cols + [target_col]].dropna(axis=1, how='all')
    if target_col not in sub.columns:
        return None
    corr_w_y = sub.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)
    top_feats = corr_w_y.index.tolist()
    top_feats = [c for c in top_feats if c != target_col][:top_n]
    if not top_feats:
        return None
    # ordenar features por corr signed con target (positivas arriba)
    signed = df[top_feats + [target_col]].corr(numeric_only=True)[target_col]
    top_feats = signed.drop(target_col).sort_values(ascending=False).index.tolist()
    cols_ordered = top_feats + [target_col]
    mat = sub[cols_ordered].corr(numeric_only=True)

    size = min(13, max(7, 0.34 * len(cols_ordered)))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(mat.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(cols_ordered)))
    ax.set_xticklabels(cols_ordered, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols_ordered)))
    ax.set_yticklabels(cols_ordered, fontsize=7)

    # marcar fila + columna del target con un borde
    tgt_idx = len(cols_ordered) - 1
    for spine_y in [tgt_idx - 0.5, tgt_idx + 0.5]:
        ax.axhline(spine_y, color='black', lw=1.3)
    for spine_x in [tgt_idx - 0.5, tgt_idx + 0.5]:
        ax.axvline(spine_x, color='black', lw=1.3)

    # anotar correlacion contra el target en la ultima columna
    for i, f in enumerate(cols_ordered):
        v = mat.loc[f, target_col]
        if not np.isfinite(v): continue
        ax.text(tgt_idx, i, f"{v:+.2f}", va='center', ha='center',
                fontsize=6, color='black' if abs(v) < 0.5 else 'white')

    ax.set_title(f'Correlacion features ↔ {target_col}  '
                 f'(top {top_n} por |corr|, ordenadas + → -)',
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('correlacion de Pearson', fontsize=9)
    fig.tight_layout()
    return _save(fig, target_dir, filename)


def plot_top_correlated_with_target(target_dir, df: pd.DataFrame,
                                    feature_cols: list, target_col: str,
                                    top_n: int = 20,
                                    filename: str = "top_correlated_features_with_VB") -> Path:
    sub = df[feature_cols + [target_col]].dropna(axis=1, how='all')
    corr = sub.corr(numeric_only=True)[target_col].drop(target_col, errors='ignore')
    corr = corr.dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(corr) + 1)))
    colors = ['#2E86AB' if v >= 0 else '#D7263D' for v in corr.values]
    ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1],
            edgecolor='k', linewidth=0.4)
    ax.axvline(0, color='k', lw=0.7)
    ax.set_xlabel(f'corr con {target_col}')
    ax.set_title(f'Top {top_n} features por |corr| con {target_col}')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return _save(fig, target_dir, filename)
