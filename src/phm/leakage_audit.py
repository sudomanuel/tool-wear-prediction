"""
leakage_audit.py — checks formales para detectar data leakage.

Cada check devuelve (status, details) donde status ∈ {PASS, FAIL, WARN}.
El resultado se guarda en outputs/metrics/leakage_checks.csv.
"""
import pandas as pd
from pathlib import Path

from .config import (
    EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL, TARGET_COLUMN,
    NON_FEATURE_COLS, PROCESSED_DATASET, SPLIT_FILE, LEAKAGE_CHECKS_CSV,
    INTERIM_AUG_DIR,
)


def _row(name, status, details):
    return {'check_name': name, 'status': status, 'details': details}


def check_one_row_per_experiment(df: pd.DataFrame):
    n = len(df)
    n_uniq = df[EXPERIMENT_ID_COL].nunique()
    if n == n_uniq:
        return _row('one_row_per_experiment', 'PASS', f'{n} filas, todas unicas por {EXPERIMENT_ID_COL}')
    return _row('one_row_per_experiment', 'FAIL',
                f'{n} filas pero solo {n_uniq} experiment_ids unicos — leakage potencial')


def check_target_unique_per_experiment(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        return _row('target_unique_per_experiment', 'FAIL', f'falta columna {TARGET_COLUMN}')
    by_exp = df.groupby(EXPERIMENT_ID_COL)[TARGET_COLUMN].nunique()
    bad = by_exp[by_exp > 1]
    if bad.empty:
        return _row('target_unique_per_experiment', 'PASS',
                    f'{len(by_exp)} experimentos con VB_um unico')
    return _row('target_unique_per_experiment', 'FAIL',
                f'experimentos con VB_um multiples: {bad.index.tolist()}')


def check_no_experiment_in_both_splits():
    if not SPLIT_FILE.exists():
        return _row('no_experiment_in_both_splits', 'WARN',
                    f'no existe {SPLIT_FILE.name}')
    rec = pd.read_csv(SPLIT_FILE)
    split_col = 'split' if 'split' in rec.columns else 'set'
    train_ids = set(rec.loc[rec[split_col] == 'train', EXPERIMENT_ID_COL].astype(int))
    test_ids  = set(rec.loc[rec[split_col] == 'test',  EXPERIMENT_ID_COL].astype(int))
    overlap = train_ids & test_ids
    if not overlap:
        return _row('no_experiment_in_both_splits', 'PASS',
                    f'train={sorted(train_ids)}, test={sorted(test_ids)}')
    return _row('no_experiment_in_both_splits', 'FAIL',
                f'experiment_ids en ambos splits: {sorted(overlap)}')


def check_id_columns_excluded(df: pd.DataFrame):
    from .dataset_builder import get_feature_columns
    feats = set(get_feature_columns(df))
    leaks = feats & {EXPERIMENT_ID_COL, TOOL_ID_COL, EXP_ORDER_COL,
                     TARGET_COLUMN, 'end_of_life', 'is_augmented'}
    if not leaks:
        return _row('id_columns_excluded', 'PASS',
                    f'features={len(feats)} sin ids ni target')
    return _row('id_columns_excluded', 'FAIL',
                f'columnas que NO deberian ser feature: {sorted(leaks)}')


def check_test_not_augmented():
    """
    Audita los CSV en data/interim/augmentation/. Las filas con
    is_augmented=True NUNCA deberian aparecer mezcladas con test.
    Como nosotros guardamos solo el TRAIN aumentado, validamos que ese
    archivo no contiene experimentos del test.
    """
    if not SPLIT_FILE.exists():
        return _row('test_not_augmented', 'WARN', 'split no existe')
    rec = pd.read_csv(SPLIT_FILE)
    split_col = 'split' if 'split' in rec.columns else 'set'
    test_ids = set(rec.loc[rec[split_col] == 'test', EXPERIMENT_ID_COL].astype(int))

    if not INTERIM_AUG_DIR.exists():
        return _row('test_not_augmented', 'PASS', 'no hay augmentation aun')
    bad = []
    for p in INTERIM_AUG_DIR.glob('train_augmented_*.csv'):
        try:
            adf = pd.read_csv(p)
        except Exception:
            continue
        if EXPERIMENT_ID_COL not in adf.columns:
            continue
        ids = set(adf[EXPERIMENT_ID_COL].astype(int).unique())
        leak = ids & test_ids
        if leak:
            bad.append(f"{p.name}:{sorted(leak)}")
    if bad:
        return _row('test_not_augmented', 'FAIL',
                    f'archivos con experimentos de test: {bad}')
    return _row('test_not_augmented', 'PASS',
                'ningun archivo de train_augmented contiene experimentos de test')


def check_augmented_rows_keep_vb(df_pre_split: pd.DataFrame):
    """
    En cada train_augmented_*.csv, las filas con is_augmented=True deben
    tener un VB_um que coincide con el del experimento original.
    """
    if not INTERIM_AUG_DIR.exists():
        return _row('augmented_rows_keep_vb', 'PASS', 'no hay augmentation aun')
    truth = dict(zip(df_pre_split[EXPERIMENT_ID_COL].astype(int),
                     df_pre_split[TARGET_COLUMN].astype(float)))
    bad = []
    for p in INTERIM_AUG_DIR.glob('train_augmented_*.csv'):
        try:
            adf = pd.read_csv(p)
        except Exception:
            continue
        if 'is_augmented' not in adf.columns:
            continue
        aug = adf[adf['is_augmented'] == True]
        for _, row in aug.iterrows():
            eid = int(row[EXPERIMENT_ID_COL])
            vb_truth = truth.get(eid)
            vb_row   = float(row[TARGET_COLUMN])
            if vb_truth is None or abs(vb_row - vb_truth) > 1e-6:
                bad.append((p.name, eid, vb_row, vb_truth))
                break  # uno por archivo basta
    if bad:
        return _row('augmented_rows_keep_vb', 'FAIL', f'desajustes: {bad[:3]}')
    return _row('augmented_rows_keep_vb', 'PASS',
                'todas las filas augmentadas conservan VB_um del experimento real')


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_all_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corre todos los checks sobre el dataset procesado y artefactos del
    pipeline existentes. Guarda outputs/metrics/leakage_checks.csv.
    """
    rows = [
        check_one_row_per_experiment(df),
        check_target_unique_per_experiment(df),
        check_no_experiment_in_both_splits(),
        check_id_columns_excluded(df),
        check_test_not_augmented(),
        check_augmented_rows_keep_vb(df),
    ]
    out = pd.DataFrame(rows)
    LEAKAGE_CHECKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LEAKAGE_CHECKS_CSV, index=False)
    return out
