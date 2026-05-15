#!/usr/bin/env python3
"""
build_dataset.py — paso 1 del pipeline.

Lee TXT raw, extrae features por contacto, mergea con target,
guarda data/processed/experiment_features.csv.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phm.config import (
    SEGMENTS_DIR, TARGET_FILE, METADATA_FILE,
    PROCESSED_DATASET, ENABLE_FREQUENCY_FEATURES,
    ensure_output_dirs,
)
from phm.dataset_builder import (
    build_dataset, write_modeling_dataset,
    write_feature_columns_manifest, write_contact_features,
)


def main():
    ensure_output_dirs()
    print("=" * 60)
    print("PASO 1 — Construccion del dataset")
    print("=" * 60)
    print(f"Segmentos : {SEGMENTS_DIR}")
    print(f"Target    : {TARGET_FILE}")
    print(f"Metadata  : {METADATA_FILE}")
    print(f"Salida    : {PROCESSED_DATASET}")
    print(f"Freq feats: {ENABLE_FREQUENCY_FEATURES}")

    df = build_dataset(
        segments_dir=SEGMENTS_DIR,
        target_file=TARGET_FILE,
        metadata_file=METADATA_FILE,
        enable_frequency=ENABLE_FREQUENCY_FEATURES,
        verbose=True,
    )
    print(f"\n[OK] experiment_features.csv shape: {df.shape}")
    print(f"[OK] columnas: {len(df.columns)}")
    print(f"[OK] experimentos: {df['experiment_id'].nunique()}")

    # Datasets derivados
    p_model = write_modeling_dataset(df)
    print(f"[OK] modeling_dataset.csv:  {p_model}")
    p_feat = write_feature_columns_manifest(df)
    print(f"[OK] feature_columns.csv:   {p_feat}")
    p_cont = write_contact_features(df)
    print(f"[OK] contact_features.csv:  {p_cont}")


if __name__ == "__main__":
    main()
