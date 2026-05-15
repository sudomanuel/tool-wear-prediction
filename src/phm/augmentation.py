"""
augmentation.py — augmentation simple a nivel feature.

Reglas estrictas:
- aplicar SOLO al train, despues del split,
- nunca tocar el test,
- VB_um NO se altera (la fila aumentada hereda el VB del experimento original),
- la fila aumentada sigue representando un experimento completo,
- columnas protegidas (ids, target, flags) jamas se perturban,
- marcar filas aumentadas con is_augmented = True.

Estrategias:
- feature_noise:     ruido gaussiano relativo al std de cada feature (1% por defecto).
- feature_scaling:   multiplicador Uniform por feature individual.
- grouped_scaling:   un mismo factor por grupo (A_p1_*, R_p3_*, ...).
"""
import re
import numpy as np
import pandas as pd
from typing import Tuple

from .config import (
    AUGMENTATION_PROTECTED_COLS, PHYSICAL_NONNEGATIVE_SUFFIXES,
    AUGMENTATION_NOISE_SIGMA, AUGMENTATION_SCALING_RANGE,
    N_AUGMENTED_PER_EXPERIMENT, RANDOM_SEED,
)


_GROUP_RE = re.compile(r'^([AR]_p\d+)_')


def _feature_cols(df: pd.DataFrame) -> list:
    """Columnas numericas que NO son protegidas."""
    out = []
    for c in df.columns:
        if c in AUGMENTATION_PROTECTED_COLS:
            continue
        if df[c].dtype == object:
            continue
        out.append(c)
    return out


def _clip_physical(row: dict, feat_cols) -> dict:
    for c in feat_cols:
        if any(c.endswith(suf) for suf in PHYSICAL_NONNEGATIVE_SUFFIXES):
            if pd.notna(row.get(c)) and row[c] < 0:
                row[c] = 0.0
    return row


# -----------------------------------------------------------------------------
# Estrategias
# -----------------------------------------------------------------------------
def _augment_feature_noise(row: pd.Series, feat_cols, stds, rng,
                           sigma: float) -> dict:
    new = row.copy().to_dict()
    for c, std_c in zip(feat_cols, stds):
        if pd.isna(new[c]):
            continue
        new[c] = float(new[c] + rng.normal(0.0, sigma * abs(std_c)))
    return new


def _augment_feature_scaling(row: pd.Series, feat_cols, rng,
                             low: float, high: float) -> dict:
    new = row.copy().to_dict()
    for c in feat_cols:
        if pd.isna(new[c]):
            continue
        new[c] = float(new[c] * rng.uniform(low, high))
    return new


def _augment_grouped_scaling(row: pd.Series, feat_cols, rng,
                             low: float, high: float) -> dict:
    """Mismo factor por grupo {A|R}_p{n}_; columnas summary -> grupo 'summary'."""
    groups: dict = {}
    for c in feat_cols:
        m = _GROUP_RE.match(c)
        key = m.group(1) if m else 'summary'
        groups.setdefault(key, []).append(c)
    factors = {g: rng.uniform(low, high) for g in groups}
    new = row.copy().to_dict()
    for g, cols in groups.items():
        f = factors[g]
        for c in cols:
            if pd.isna(new[c]):
                continue
            new[c] = float(new[c] * f)
    return new


# -----------------------------------------------------------------------------
# API publica
# -----------------------------------------------------------------------------
def augment_train(train_df: pd.DataFrame,
                  strategy: str = 'feature_noise',
                  n_augmented: int = N_AUGMENTED_PER_EXPERIMENT,
                  noise_sigma: float = AUGMENTATION_NOISE_SIGMA,
                  scaling_range: Tuple[float, float] = AUGMENTATION_SCALING_RANGE,
                  seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Devuelve un dataframe que contiene:
      - las filas originales (is_augmented=False),
      - las filas aumentadas (is_augmented=True).

    Nunca toca columnas protegidas (incluyendo VB_um).
    """
    train_df = train_df.copy()
    if 'is_augmented' not in train_df.columns:
        train_df['is_augmented'] = False

    if strategy == 'none' or n_augmented <= 0:
        return train_df

    feat_cols = _feature_cols(train_df)
    if not feat_cols:
        return train_df

    stds = train_df[feat_cols].std(ddof=0).fillna(0.0).values
    rng  = np.random.default_rng(seed)
    low, high = scaling_range

    augmented_rows = []
    for _, row in train_df.iterrows():
        for _ in range(n_augmented):
            if strategy == 'feature_noise':
                new = _augment_feature_noise(row, feat_cols, stds, rng, noise_sigma)
            elif strategy == 'feature_scaling':
                new = _augment_feature_scaling(row, feat_cols, rng, low, high)
            elif strategy == 'grouped_scaling':
                new = _augment_grouped_scaling(row, feat_cols, rng, low, high)
            else:
                raise ValueError(f"Estrategia desconocida: {strategy}")
            new = _clip_physical(new, feat_cols)
            new['is_augmented'] = True
            augmented_rows.append(new)

    aug_df = pd.DataFrame(augmented_rows, columns=train_df.columns)
    out = pd.concat([train_df, aug_df], ignore_index=True)
    return out
