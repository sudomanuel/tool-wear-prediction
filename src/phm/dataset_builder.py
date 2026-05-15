"""
dataset_builder.py — construye experiment_features.csv.

REGLA CRITICA: una fila = un experimento completo.
VB_um se mide una sola vez por experimento (al final, despues de los 6
contactos). NUNCA se repite VB_um por contacto.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .config import (
    N_CONTACTS, AXIAL_PREFIX, ROT_PREFIX,
    ENABLE_FREQUENCY_FEATURES, MIN_SAMPLES_FOR_FFT,
    SEGMENTS_DIR, TARGET_FILE, METADATA_FILE,
    EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL, TARGET_COLUMN,
    NON_FEATURE_COLS, PROCESSED_DATASET, PROCESSED_DIR,
    CONTACT_FEATURES, MODELING_DATASET, FEATURE_COLUMNS_CSV, INTERIM_DIR,
    METRICS_DIR,
)
from .filename_parser import scan_segments
from .data_loader import load_signal, load_target_csv
from .preprocessing import preprocess_signal
from .feature_extraction import extract_all_features, all_feature_names


# -----------------------------------------------------------------------------
# Fila por experimento
# -----------------------------------------------------------------------------
def build_experiment_row(experiment_id: int,
                         seg_paths: dict,
                         enable_frequency: bool,
                         verbose: bool = False) -> dict:
    """
    seg_paths = {(dir_code, contact_id): Path}
    Concatena features de los 12 segmentos en una sola fila plana.
    """
    row = {EXPERIMENT_ID_COL: experiment_id}
    rms_per_dir   = {AXIAL_PREFIX: [], ROT_PREFIX: []}
    energy_per_dir = {AXIAL_PREFIX: [], ROT_PREFIX: []}

    for cid in range(1, N_CONTACTS + 1):
        for dir_code in (AXIAL_PREFIX, ROT_PREFIX):
            tag = f"{dir_code}_p{cid}"
            path = seg_paths.get((dir_code, cid))
            feats = None
            if path is not None:
                df = load_signal(path)
                if df is not None:
                    df_c, fs = preprocess_signal(df, center=True)
                    if df_c is not None and not df_c.empty:
                        feats = extract_all_features(
                            df_c['vibration_value'].values,
                            df_c['timestamp'].values,
                            sampling_rate_hz=fs,
                            enable_frequency=enable_frequency,
                            min_samples_fft=MIN_SAMPLES_FOR_FFT,
                        )
            if feats is None:
                feats = {f: float('nan') for f in all_feature_names(enable_frequency)}
                if verbose:
                    print(f"   [{tag}] FALTANTE -> NaN")

            for k, v in feats.items():
                row[f"{tag}_{k}"] = v

            rms_per_dir[dir_code].append(feats.get('rms', float('nan')))
            energy_per_dir[dir_code].append(feats.get('energy', float('nan')))

    # Agregadas multi-contacto
    for dir_code in (AXIAL_PREFIX, ROT_PREFIX):
        rms_arr = np.array(rms_per_dir[dir_code], dtype=float)
        erg_arr = np.array(energy_per_dir[dir_code], dtype=float)
        row[f'{dir_code}_rms_mean_6_contacts']     = float(np.nanmean(rms_arr))
        row[f'{dir_code}_rms_std_6_contacts']      = float(np.nanstd(rms_arr))
        row[f'{dir_code}_energy_total_6_contacts'] = float(np.nansum(erg_arr))

    A_rms = np.array(rms_per_dir[AXIAL_PREFIX], dtype=float)
    R_rms = np.array(rms_per_dir[ROT_PREFIX],   dtype=float)
    A_erg = np.array(energy_per_dir[AXIAL_PREFIX], dtype=float)
    R_erg = np.array(energy_per_dir[ROT_PREFIX],   dtype=float)

    row['total_energy_6_contacts'] = float(np.nansum(np.concatenate([A_erg, R_erg])))
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio_rms = np.where(R_rms > 0, A_rms / R_rms, np.nan)
        ratio_erg = np.where(R_erg > 0, A_erg / R_erg, np.nan)
    row['A_to_R_rms_ratio']    = float(np.nanmean(ratio_rms))
    row['A_to_R_energy_ratio'] = float(np.nanmean(ratio_erg))
    return row


# -----------------------------------------------------------------------------
# Dataset completo
# -----------------------------------------------------------------------------
def build_dataset(segments_dir: Path = SEGMENTS_DIR,
                  target_file: Path = TARGET_FILE,
                  metadata_file: Optional[Path] = METADATA_FILE,
                  enable_frequency: bool = ENABLE_FREQUENCY_FEATURES,
                  verbose: bool = True) -> pd.DataFrame:
    segments_dir = Path(segments_dir)
    if not segments_dir.exists():
        raise FileNotFoundError(f"Falta carpeta de segmentos: {segments_dir}")

    seg_index = scan_segments(segments_dir)
    if not seg_index:
        raise ValueError(f"No se encontraron segmentos parseables en {segments_dir}")

    target_df = load_target_csv(target_file)
    if TARGET_COLUMN not in target_df.columns:
        raise ValueError(f"Target file no contiene columna {TARGET_COLUMN}")
    target_df[EXPERIMENT_ID_COL] = pd.to_numeric(target_df[EXPERIMENT_ID_COL], errors='coerce').astype('Int64')
    target_df[TARGET_COLUMN]     = pd.to_numeric(target_df[TARGET_COLUMN], errors='coerce')
    target_df = target_df.dropna(subset=[EXPERIMENT_ID_COL, TARGET_COLUMN])
    target_df[EXPERIMENT_ID_COL] = target_df[EXPERIMENT_ID_COL].astype(int)

    # Solo construimos features para experimentos con target conocido
    valid_ids = sorted(set(seg_index.keys()) & set(target_df[EXPERIMENT_ID_COL].tolist()))
    if not valid_ids:
        raise ValueError("Ningun experiment_id tiene a la vez segmentos y target.")

    if verbose:
        print(f"[BUILD] experimentos a procesar: {valid_ids}")

    rows = []
    for eid in valid_ids:
        if verbose:
            print(f"  exp {eid} ...")
        rows.append(build_experiment_row(eid, seg_index[eid],
                                          enable_frequency=enable_frequency,
                                          verbose=verbose))

    feats_df = pd.DataFrame(rows)

    # Merge con target (1:1 por experiment_id)
    dataset = feats_df.merge(target_df, on=EXPERIMENT_ID_COL, how='inner')

    # Merge con metadata si existe
    if metadata_file is not None and Path(metadata_file).exists():
        try:
            meta_df = pd.read_csv(metadata_file)
            meta_df.columns = [c.strip() for c in meta_df.columns]
            # Evitar columnas duplicadas
            keep = [c for c in meta_df.columns
                    if c == EXPERIMENT_ID_COL or c not in dataset.columns]
            if EXPERIMENT_ID_COL in meta_df.columns:
                dataset = dataset.merge(meta_df[keep], on=EXPERIMENT_ID_COL, how='left')
        except Exception as exc:
            warnings.warn(f"[META] no se pudo mergear metadata: {exc}")

    # Orden de columnas: id, metadata, features, target al final
    front = [c for c in (EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL) if c in dataset.columns]
    back  = [TARGET_COLUMN]
    other = [c for c in dataset.columns if c not in front + back]
    dataset = dataset[front + other + back]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(PROCESSED_DATASET, index=False)
    if verbose:
        print(f"[BUILD] dataset guardado: {PROCESSED_DATASET}  shape={dataset.shape}")
    return dataset


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list:
    """Devuelve columnas numericas que son features ML legitimas."""
    out = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if df[c].dtype == object:
            continue
        out.append(c)
    return out


def get_X_y_groups(df: pd.DataFrame):
    feats = get_feature_columns(df)
    X = df[feats].values.astype(float)
    y = df[TARGET_COLUMN].values.astype(float)
    groups = df[EXPERIMENT_ID_COL].values
    return X, y, groups, feats


# -----------------------------------------------------------------------------
# Datasets derivados (largos / interim)
# -----------------------------------------------------------------------------
def build_contact_features_long(experiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del dataset wide (1 fila = 1 experimento), construye una version
    long con una fila por (experiment_id, direction, contact_id).
    Util para debugging / exploracion; NO se usa para entrenar.
    """
    rows = []
    for _, exp in experiment_df.iterrows():
        eid = int(exp[EXPERIMENT_ID_COL])
        for dir_code in (AXIAL_PREFIX, ROT_PREFIX):
            for cid in range(1, N_CONTACTS + 1):
                prefix = f"{dir_code}_p{cid}_"
                row = {
                    EXPERIMENT_ID_COL: eid,
                    'direction': 'axial' if dir_code == AXIAL_PREFIX else 'rotational',
                    'contact_id': cid,
                }
                for c in experiment_df.columns:
                    if c.startswith(prefix):
                        # quitar el prefijo en la version long
                        row[c[len(prefix):]] = exp[c]
                rows.append(row)
    return pd.DataFrame(rows)


def write_modeling_dataset(experiment_df: pd.DataFrame) -> Path:
    """
    Guarda data/processed/modeling_dataset.csv con columnas:
      - first:   experiment_id (id, NO feature)
      - middle:  features numericas
      - last:    target (VB_um)
    Excluye explicitamente columnas no-feature.
    """
    feat_cols = get_feature_columns(experiment_df)
    cols = [EXPERIMENT_ID_COL] + feat_cols + [TARGET_COLUMN]
    cols = [c for c in cols if c in experiment_df.columns]
    out = experiment_df[cols].copy()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(MODELING_DATASET, index=False)
    return MODELING_DATASET


def write_feature_columns_manifest(experiment_df: pd.DataFrame) -> Path:
    """
    Lista todas las columnas del dataset wide e indica si se usan como
    feature ML o no, con la razon.
    """
    feat_cols = set(get_feature_columns(experiment_df))
    rows = []
    for c in experiment_df.columns:
        dtype = str(experiment_df[c].dtype)
        if c in feat_cols:
            rows.append({
                'feature_name': c, 'dtype': dtype,
                'used_in_model': True, 'reason_if_excluded': '',
            })
        else:
            if c == TARGET_COLUMN:
                reason = 'target'
            elif c == EXPERIMENT_ID_COL:
                reason = 'identifier (experiment_id)'
            elif c == TOOL_ID_COL:
                reason = 'identifier (tool_id, no numerico)'
            elif c == EXP_ORDER_COL:
                reason = 'order proxy temporal — excluido por defecto'
            elif c == 'end_of_life':
                reason = 'flag derivado del experimento, no feature de senal'
            elif c == 'is_augmented':
                reason = 'flag de augmentation'
            elif experiment_df[c].dtype == object:
                reason = 'categorica no codificada'
            else:
                reason = 'excluida por NON_FEATURE_COLS'
            rows.append({
                'feature_name': c, 'dtype': dtype,
                'used_in_model': False, 'reason_if_excluded': reason,
            })
    df = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_COLUMNS_CSV, index=False)
    return FEATURE_COLUMNS_CSV


def write_contact_features(experiment_df: pd.DataFrame) -> Path:
    long_df = build_contact_features_long(experiment_df)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(CONTACT_FEATURES, index=False)
    return CONTACT_FEATURES
