"""
splitting.py — splits a nivel experiment_id (anti-leakage).

- holdout_split: 8/2 deterministico via GroupShuffleSplit.
- loeo_iter:     Leave-One-Experiment-Out (10 folds para 10 experimentos).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterator
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut

from .config import (
    RANDOM_SEED, TEST_SIZE, SPLIT_FILE, SPLITS_DIR,
    LOEO_FOLDS_FILE, EXPERIMENT_ID_COL, TARGET_COLUMN,
    TOOL_ID_COL, EXP_ORDER_COL,
)


def holdout_split(df: pd.DataFrame,
                  group_col: str,
                  test_size: float = TEST_SIZE,
                  seed: int = RANDOM_SEED,
                  save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split por grupo (experiment_id). Deterministico.
    Devuelve (train_df, test_df). Si save=True, escribe outputs/splits/.
    """
    groups = df[group_col].values
    idx = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(idx, groups=groups))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df  = df.iloc[test_idx].copy().reset_index(drop=True)
    if save:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        extra_cols = [c for c in (TARGET_COLUMN, TOOL_ID_COL, EXP_ORDER_COL)
                      if c in df.columns]
        rec_rows = []
        for sub, label in [(train_df, 'train'), (test_df, 'test')]:
            for _, r in sub.iterrows():
                row = {group_col: int(r[group_col]), 'split': label}
                for c in extra_cols:
                    row[c] = r[c]
                rec_rows.append(row)
        pd.DataFrame(rec_rows).to_csv(SPLIT_FILE, index=False)
    return train_df, test_df


def write_loeo_folds(df: pd.DataFrame, group_col: str = EXPERIMENT_ID_COL):
    """Lista cada fold LOEO: fold_id, test_experiment_id, train_experiment_ids."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    groups = df[group_col].astype(int).values
    for i, test_id in enumerate(sorted(set(groups)), start=1):
        train_ids = sorted(int(g) for g in groups if int(g) != int(test_id))
        rows.append({
            'fold_id': i,
            'test_experiment_id': int(test_id),
            'train_experiment_ids': ';'.join(str(x) for x in train_ids),
        })
    pd.DataFrame(rows).to_csv(LOEO_FOLDS_FILE, index=False)


def load_holdout_split(df: pd.DataFrame, group_col: str
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el split guardado y reconstruye (train_df, test_df)."""
    if not SPLIT_FILE.exists():
        return holdout_split(df, group_col=group_col, save=True)
    rec = pd.read_csv(SPLIT_FILE)
    # Soporta tanto el nombre antiguo 'set' como el nuevo 'split'
    split_col = 'split' if 'split' in rec.columns else 'set'
    train_ids = set(rec.loc[rec[split_col] == 'train', group_col].astype(int))
    test_ids  = set(rec.loc[rec[split_col] == 'test',  group_col].astype(int))
    train_df = df[df[group_col].astype(int).isin(train_ids)].reset_index(drop=True)
    test_df  = df[df[group_col].astype(int).isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def loeo_iter(df: pd.DataFrame, group_col: str
              ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Genera (train_df, test_df) para Leave-One-Experiment-Out.
    Con 10 experimentos => 10 iteraciones, test = 1 experimento, train = 9.
    """
    groups = df[group_col].values
    logo = LeaveOneGroupOut()
    idx = np.arange(len(df))
    for tr_idx, te_idx in logo.split(idx, groups=groups):
        yield (df.iloc[tr_idx].reset_index(drop=True),
               df.iloc[te_idx].reset_index(drop=True))
