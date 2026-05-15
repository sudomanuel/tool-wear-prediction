"""
data_quality.py — inventario de archivos raw, deteccion de archivos
faltantes, plots de calidad del dataset.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

from .config import (
    N_CONTACTS, AXIAL_PREFIX, ROT_PREFIX,
    SEGMENTS_DIR, TARGET_FILE, EXPERIMENT_ID_COL, EXP_ORDER_COL, TARGET_COLUMN,
    DATA_INVENTORY_CSV, MISSING_SEGMENTS, FIG_DATA_QUALITY, FIG_SIGNALS,
    FIGURE_DPI, FIGURE_FORMAT,
)
from .filename_parser import scan_segments
from .data_loader import load_signal, load_target_csv


def _direction_full(code: str) -> str:
    return 'axial' if code == AXIAL_PREFIX else 'rotational'


def build_inventory(segments_dir: Path = SEGMENTS_DIR,
                    target_file: Path = TARGET_FILE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recorre data/raw/segments/, carga (parcialmente) cada TXT y construye:
      - inventario: una fila por archivo TXT presente
      - missing:    una fila por archivo TXT esperado que NO existe
    """
    target_df = load_target_csv(target_file)
    expected_exp_ids = sorted(int(x) for x in target_df[EXPERIMENT_ID_COL].dropna().tolist())

    seg_index = scan_segments(segments_dir)

    # --- inventario ---
    inv_rows = []
    for eid, files in seg_index.items():
        for (dir_code, cid), path in sorted(files.items()):
            df = load_signal(path)
            ok = df is not None and not df.empty
            n_rows  = int(len(df)) if ok else 0
            has_ts  = bool(ok and df['timestamp'].notna().all())
            has_vib = bool(ok and df['vibration_value'].notna().all())
            status = 'OK' if (ok and has_ts and has_vib) else 'PROBLEM'
            inv_rows.append({
                'filename': path.name,
                'path': str(path),
                'experiment_id': int(eid),
                'direction': _direction_full(dir_code),
                'contact_id': int(cid),
                'exists': True,
                'n_rows': n_rows,
                'has_valid_timestamp': has_ts,
                'has_valid_vibration_value': has_vib,
                'status': status,
            })
    inv_df = pd.DataFrame(inv_rows).sort_values(
        ['experiment_id', 'direction', 'contact_id']
    ).reset_index(drop=True)

    # --- missing ---
    miss_rows = []
    for eid in expected_exp_ids:
        for dir_code in (AXIAL_PREFIX, ROT_PREFIX):
            for cid in range(1, N_CONTACTS + 1):
                if (dir_code, cid) in seg_index.get(eid, {}):
                    continue
                miss_rows.append({
                    'experiment_id': int(eid),
                    'expected_file': f"{dir_code}{eid}_p{cid}.txt",
                    'direction': _direction_full(dir_code),
                    'contact_id': int(cid),
                    'status': 'MISSING',
                })
    miss_df = pd.DataFrame(miss_rows).sort_values(
        ['experiment_id', 'direction', 'contact_id']
    ).reset_index(drop=True) if miss_rows else pd.DataFrame(
        columns=['experiment_id', 'expected_file', 'direction', 'contact_id', 'status']
    )

    DATA_INVENTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(DATA_INVENTORY_CSV, index=False)
    miss_df.to_csv(MISSING_SEGMENTS, index=False)
    return inv_df, miss_df


# -----------------------------------------------------------------------------
# Plots de calidad
# -----------------------------------------------------------------------------
def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_raw_file_count(inv_df: pd.DataFrame):
    if inv_df.empty:
        return
    counts = inv_df.groupby('experiment_id').size()
    expected = N_CONTACTS * 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values, color='#2E86AB',
           edgecolor='k', linewidth=0.5)
    ax.axhline(expected, color='red', linestyle='--',
               label=f'esperado = {expected}')
    ax.set_xlabel('experiment_id')
    ax.set_ylabel('# archivos TXT presentes')
    ax.set_title('Archivos raw por experimento')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, FIG_DATA_QUALITY / f"raw_file_count_by_experiment.{FIGURE_FORMAT}")


def plot_missing_segments(miss_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    if miss_df.empty:
        ax.text(0.5, 0.5, 'Sin archivos faltantes',
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
    else:
        counts = miss_df.groupby('experiment_id').size()
        ax.bar(counts.index.astype(str), counts.values,
               color='#D7263D', edgecolor='k', linewidth=0.5)
        ax.set_xlabel('experiment_id')
        ax.set_ylabel('# archivos esperados que faltan')
        ax.set_title('Archivos faltantes por experimento')
        ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, FIG_DATA_QUALITY / f"missing_segments_by_experiment.{FIGURE_FORMAT}")


def plot_vb_distribution(target_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(target_df[TARGET_COLUMN].values, bins=10,
            color='#2E86AB', edgecolor='k', alpha=0.85)
    ax.set_xlabel(f'{TARGET_COLUMN} (µm)')
    ax.set_ylabel('frecuencia')
    ax.set_title(f'Distribucion de {TARGET_COLUMN}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, FIG_DATA_QUALITY / f"vb_distribution.{FIGURE_FORMAT}")


def plot_vb_vs_order(target_df: pd.DataFrame):
    if EXP_ORDER_COL not in target_df.columns:
        return
    sub = target_df.dropna(subset=[EXP_ORDER_COL, TARGET_COLUMN]).copy()
    sub = sub.sort_values(EXP_ORDER_COL)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sub[EXP_ORDER_COL], sub[TARGET_COLUMN], 'o-',
            color='#2E86AB', markeredgecolor='k')
    ax.set_xlabel(EXP_ORDER_COL)
    ax.set_ylabel(f'{TARGET_COLUMN} (µm)')
    ax.set_title(f'{TARGET_COLUMN} vs experiment_order')
    ax.grid(True, alpha=0.3)
    for _, row in sub.iterrows():
        ax.annotate(int(row[EXPERIMENT_ID_COL]),
                    (row[EXP_ORDER_COL], row[TARGET_COLUMN]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    fig.tight_layout()
    _save(fig, FIG_DATA_QUALITY / f"vb_vs_experiment_order.{FIGURE_FORMAT}")


def make_all_quality_plots(inv_df: pd.DataFrame, miss_df: pd.DataFrame,
                            target_df: pd.DataFrame):
    plot_raw_file_count(inv_df)
    plot_missing_segments(miss_df)
    plot_vb_distribution(target_df)
    plot_vb_vs_order(target_df)


# -----------------------------------------------------------------------------
# Signal examples
# -----------------------------------------------------------------------------
def plot_signal_examples(segments_dir: Path = SEGMENTS_DIR,
                         example_exp_id: int = 66):
    """Plot rapido de 2 senales individuales + un grid de los 6 contactos."""
    from .preprocessing import preprocess_signal

    # ejemplo 1: A_p1
    p_a = segments_dir / f"A{example_exp_id}_p1.txt"
    p_r = segments_dir / f"R{example_exp_id}_p1.txt"

    for path, suf in [(p_a, 'A_p1'), (p_r, 'R_p1')]:
        df = load_signal(path)
        if df is None:
            continue
        df_c, _ = preprocess_signal(df, center=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df_c['timestamp'], df_c['vibration_value'], lw=0.5,
                color='#1F77B4')
        ax.set_title(f'Senal exp {example_exp_id} {suf}')
        ax.set_xlabel('timestamp')
        ax.set_ylabel('vibracion (centrada)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, FIG_SIGNALS / f"example_signal_{suf}.{FIGURE_FORMAT}")

    # grid 6 contactos axiales
    fig, axes = plt.subplots(2, 3, figsize=(13, 5), sharex=False, sharey=False)
    for i, ax in enumerate(axes.flat, start=1):
        path = segments_dir / f"A{example_exp_id}_p{i}.txt"
        df = load_signal(path)
        if df is None:
            ax.set_title(f'A_p{i} (faltante)')
            ax.set_axis_off()
            continue
        df_c, _ = preprocess_signal(df, center=True)
        ax.plot(df_c['timestamp'], df_c['vibration_value'], lw=0.4,
                color='#1F77B4')
        ax.set_title(f'A_p{i}')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Senales axiales — experimento {example_exp_id}', y=1.02)
    fig.tight_layout()
    _save(fig, FIG_SIGNALS / f"example_contacts_grid_exp{example_exp_id}.{FIGURE_FORMAT}")
