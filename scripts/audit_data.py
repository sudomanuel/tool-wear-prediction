#!/usr/bin/env python3
"""
audit_data.py — paso 0 del pipeline.

Inventario de archivos raw, deteccion de archivos faltantes y plots de
calidad. Solo lectura: no modifica data/raw/.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm.config import (
    SEGMENTS_DIR, TARGET_FILE,
    DATA_INVENTORY_CSV, MISSING_SEGMENTS,
    ensure_output_dirs,
)

EXAMPLE_EXP_ID = 66
from phm.data_quality import (
    build_inventory, make_all_quality_plots, plot_signal_examples,
)
from phm.data_loader import load_target_csv


def main():
    ensure_output_dirs()
    print("=" * 60)
    print("PASO 0 — Auditoria de datos raw")
    print("=" * 60)
    inv_df, miss_df = build_inventory(SEGMENTS_DIR, TARGET_FILE)
    print(f"[INV] archivos encontrados: {len(inv_df)}")
    print(f"[INV] archivos faltantes  : {len(miss_df)}")
    if not miss_df.empty:
        print(miss_df.to_string(index=False))

    target_df = load_target_csv(TARGET_FILE)
    make_all_quality_plots(inv_df, miss_df, target_df)
    plot_signal_examples(SEGMENTS_DIR, example_exp_id=66)

    print(f"[OK] {DATA_INVENTORY_CSV}")
    print(f"[OK] {MISSING_SEGMENTS}")


if __name__ == "__main__":
    main()
