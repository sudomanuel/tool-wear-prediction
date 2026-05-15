"""
preprocessing.py — pasos minimos por senal antes de extraer features.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def preprocess_signal(df: pd.DataFrame,
                      center: bool = True) -> Tuple[pd.DataFrame, Optional[float]]:
    """
    Limpia y centra la senal. Devuelve (df_limpio, sampling_rate_hz).
    El sampling rate se estima del timestamp si hay > 1 muestra.
    """
    if df is None or df.empty:
        return df, None

    df = df.dropna(subset=['timestamp', 'vibration_value']).copy()
    df = df.drop_duplicates(subset='timestamp', keep='first')
    df = df.sort_values('timestamp').reset_index(drop=True)

    if len(df) <= 1:
        return df, None

    # Sampling rate estimado: 1 / mediana del delta
    dt = np.diff(df['timestamp'].values)
    dt = dt[dt > 0]
    fs = float(1.0 / np.median(dt)) if len(dt) > 0 else None

    if center:
        df['vibration_value'] = df['vibration_value'] - df['vibration_value'].mean()

    return df, fs
