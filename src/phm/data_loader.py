"""
data_loader.py — carga robusta de TXT con auto-deteccion de separador.
"""
import csv
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def _detect_sep(path: Path) -> str:
    """Detecta separador (,/;/\t/espacios) sin asumir nada."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            sample = ''.join(f.readline() for _ in range(20))
        try:
            return csv.Sniffer().sniff(sample, delimiters=',;\t').delimiter
        except csv.Error:
            pass
        counts = {s: sample.count(s) for s in (',', ';', '\t')}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else r'\s+'
    except Exception:
        return r'\s+'


def load_signal(path: Path) -> Optional[pd.DataFrame]:
    """
    Devuelve DataFrame con columnas ['timestamp', 'vibration_value'] o None
    si el archivo no se pudo parsear. No lanza excepciones por archivos
    aislados — escribe warning y sigue.
    """
    path = Path(path)
    if not path.exists():
        return None
    sep = _detect_sep(path)
    try:
        df = pd.read_csv(path, sep=sep, header=None, engine='python',
                         on_bad_lines='skip')
    except Exception as exc:
        warnings.warn(f"[LOADER] no se pudo leer {path.name}: {exc}")
        return None
    if df.shape[1] < 2:
        # reintento con whitespace
        try:
            df = pd.read_csv(path, sep=r'\s+', header=None,
                             engine='python', on_bad_lines='skip')
        except Exception:
            return None
    if df.shape[1] < 2 or df.empty:
        return None
    df = df.iloc[:, :2].copy()
    df.columns = ['timestamp', 'vibration_value']
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(inplace=True)
    if df.empty:
        return None
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_target_csv(path: Path) -> pd.DataFrame:
    """Carga vb_targets.csv. Soporta BOM y separadores comunes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")
    for enc in ('utf-8-sig', 'utf-8', 'latin-1'):
        try:
            for sep in (',', ';', '\t'):
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] >= 2:
                    df.columns = [str(c).strip().lstrip('﻿') for c in df.columns]
                    return df
        except Exception:
            continue
    raise ValueError(f"No se pudo leer {path}")
