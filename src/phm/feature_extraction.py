"""
feature_extraction.py — features estadisticas + (opcional) frecuenciales
por segmento de contacto.

Time-domain (13): mean, std, rms, max, min, peak_to_peak, skewness, kurtosis,
                  crest_factor, energy, absolute_mean, duration_s, n_samples
Freq-domain (3):  dominant_freq_hz, spectral_energy, spectral_centroid_hz
"""
import warnings
import numpy as np
from typing import Optional
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurtosis


TIME_FEATURE_NAMES = [
    'mean', 'std', 'rms', 'max', 'min', 'peak_to_peak',
    'skewness', 'kurtosis', 'crest_factor', 'energy',
    'absolute_mean', 'duration_s', 'n_samples',
]
FREQ_FEATURE_NAMES = [
    'dominant_freq_hz', 'spectral_energy', 'spectral_centroid_hz',
]


def _nan_dict(keys):
    return {k: float('nan') for k in keys}


def extract_time_features(signal: np.ndarray, timestamps: np.ndarray) -> dict:
    n = len(signal)
    if n == 0:
        return _nan_dict(TIME_FEATURE_NAMES)
    duration = float(timestamps[-1] - timestamps[0]) if n > 1 else 0.0
    mean_v   = float(np.mean(signal))
    std_v    = float(np.std(signal, ddof=0))
    rms_v    = float(np.sqrt(np.mean(signal ** 2)))
    max_v    = float(np.max(signal))
    min_v    = float(np.min(signal))
    p2p_v    = max_v - min_v
    abs_mean = float(np.mean(np.abs(signal)))
    kurt_v   = float(scipy_kurtosis(signal, fisher=True, bias=True)) if n >= 4 else float('nan')
    skew_v   = float(scipy_skew(signal, bias=True)) if n >= 3 else float('nan')
    peak_abs = float(np.max(np.abs(signal)))
    crest_v  = (peak_abs / rms_v) if rms_v > 0 else float('nan')
    dt       = duration / (n - 1) if (n > 1 and duration > 0) else 1.0
    energy_v = float(np.sum(signal ** 2) * dt)
    return {
        'mean': mean_v, 'std': std_v, 'rms': rms_v,
        'max': max_v, 'min': min_v, 'peak_to_peak': p2p_v,
        'skewness': skew_v, 'kurtosis': kurt_v, 'crest_factor': crest_v,
        'energy': energy_v, 'absolute_mean': abs_mean,
        'duration_s': duration, 'n_samples': n,
    }


def extract_frequency_features(signal: np.ndarray,
                               sampling_rate_hz: float,
                               min_samples: int = 64) -> dict:
    n = len(signal)
    if n < min_samples or sampling_rate_hz is None or sampling_rate_hz <= 0:
        return _nan_dict(FREQ_FEATURE_NAMES)
    try:
        fft_v = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
        psd   = np.abs(fft_v) ** 2
        dom_freq = float(freqs[int(np.argmax(psd))])
        spec_e   = float(np.sum(psd))
        centroid = float(np.sum(freqs * psd) / spec_e) if spec_e > 0 else float('nan')
        return {
            'dominant_freq_hz': dom_freq,
            'spectral_energy': spec_e,
            'spectral_centroid_hz': centroid,
        }
    except Exception as exc:
        warnings.warn(f"[FFT] fallo: {exc}")
        return _nan_dict(FREQ_FEATURE_NAMES)


def extract_all_features(signal: np.ndarray,
                         timestamps: np.ndarray,
                         sampling_rate_hz: Optional[float],
                         enable_frequency: bool = True,
                         min_samples_fft: int = 64) -> dict:
    feats = extract_time_features(signal, timestamps)
    if enable_frequency and sampling_rate_hz is not None and sampling_rate_hz > 0:
        feats.update(extract_frequency_features(signal, sampling_rate_hz,
                                                min_samples=min_samples_fft))
    else:
        feats.update(_nan_dict(FREQ_FEATURE_NAMES))
    return feats


def all_feature_names(include_frequency: bool = True) -> list:
    return list(TIME_FEATURE_NAMES) + (list(FREQ_FEATURE_NAMES) if include_frequency else [])
