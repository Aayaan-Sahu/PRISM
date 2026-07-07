"""Waveform-level augmentations for the on-the-fly mixer. All numpy float32,
CPU, per-sample (runs inside DataLoader workers)."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, sosfilt

try:
    import pyloudnorm as _pyln
    _METER_CACHE: dict[int, "_pyln.Meter"] = {}
except ImportError:  # RMS fallback below
    _pyln = None

EPS = 1e-8


# --------------------------------------------------------------------------
# Audio loading
# --------------------------------------------------------------------------
def load_segment(path: str, n_samples: int, rng, expected_sr: int = 16000) -> np.ndarray:
    """Random n_samples-long mono segment via seek (never decodes whole long
    files). Files shorter than n_samples are tiled."""
    with sf.SoundFile(path) as f:
        if f.samplerate != expected_sr:
            raise ValueError(f"{path}: sr={f.samplerate}, expected {expected_sr} "
                             "(run data/build_manifests.py to convert)")
        total = f.frames
        if total <= n_samples:
            data = f.read(dtype="float32", always_2d=True)
        else:
            start = int(rng.integers(0, total - n_samples))
            f.seek(start)
            data = f.read(n_samples, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if len(x) < n_samples:
        reps = int(np.ceil(n_samples / max(len(x), 1)))
        x = np.tile(x, reps)[:n_samples]
    return np.ascontiguousarray(x, dtype=np.float32)


def load_rir(path: str, sr: int = 16000, max_s: float = 0.6) -> np.ndarray:
    """Load a RIR, align its direct path to t~0 (so reverberant mixture and
    early-reverb ground truth stay time-aligned), truncate, peak-normalize."""
    rir, file_sr = sf.read(path, dtype="float32", always_2d=True)
    rir = rir.mean(axis=1)
    if file_sr != sr:
        # cheap linear resample; RIRs tolerate this fine
        n_out = int(round(len(rir) * sr / file_sr))
        rir = np.interp(
            np.linspace(0, len(rir) - 1, n_out), np.arange(len(rir)), rir
        ).astype(np.float32)
    onset = int(np.argmax(np.abs(rir)))
    rir = rir[max(0, onset - 8):]
    rir = rir[: int(max_s * sr)]
    peak = np.max(np.abs(rir))
    return rir / peak if peak > EPS else rir


def early_rir(rir: np.ndarray, sr: int, early_ms: float) -> np.ndarray:
    return rir[: int(early_ms / 1000.0 * sr)]


def apply_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return fftconvolve(x, rir)[: len(x)].astype(np.float32)


# --------------------------------------------------------------------------
# Levels
# --------------------------------------------------------------------------
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS))


def gain_for_ratio(ref: np.ndarray, other: np.ndarray, ratio_db: float) -> float:
    """Gain for `other` so that 20log10(rms(ref)/rms(g*other)) == ratio_db."""
    return rms(ref) / (rms(other) * (10.0 ** (ratio_db / 20.0)) + EPS)


def lufs_gain(x: np.ndarray, target_lufs: float, sr: int) -> float:
    """Linear gain that brings x to target integrated loudness. Falls back to
    RMS-dBFS matching when pyloudnorm is unavailable or the clip defeats it."""
    if _pyln is not None and len(x) >= sr:  # pyloudnorm needs >= 400ms; be safe
        meter = _METER_CACHE.get(sr)
        if meter is None:
            meter = _METER_CACHE[sr] = _pyln.Meter(sr)
        loud = meter.integrated_loudness(x.astype(np.float64))
        if np.isfinite(loud):
            return float(10.0 ** ((target_lufs - loud) / 20.0))
    cur_db = 20.0 * np.log10(rms(x))
    return float(10.0 ** ((target_lufs - cur_db) / 20.0))


# --------------------------------------------------------------------------
# Channel simulation
# --------------------------------------------------------------------------
def _peaking_sos(f0: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """RBJ audio-EQ-cookbook peaking biquad, as a (1, 6) sos row."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cosw = np.cos(w0)
    b = np.array([1 + alpha * a, -2 * cosw, 1 - alpha * a])
    aa = np.array([1 + alpha / a, -2 * cosw, 1 - alpha / a])
    return np.concatenate([b / aa[0], aa / aa[0]]).reshape(1, 6)


def random_eq(x: np.ndarray, rng, sr: int) -> np.ndarray:
    for _ in range(int(rng.integers(1, 4))):
        f0 = float(np.exp(rng.uniform(np.log(100.0), np.log(7000.0))))
        sos = _peaking_sos(f0, float(rng.uniform(-6, 6)), float(rng.uniform(0.5, 2.0)), sr)
        x = sosfilt(sos, x).astype(np.float32)
    return x


def spectral_tilt(x: np.ndarray, rng) -> np.ndarray:
    """Mild broadband tilt: blend in a one-pole low- or high-passed copy."""
    a = float(rng.uniform(0.85, 0.98))
    sos = np.array([[1 - a, 0.0, 0.0, 1.0, -a, 0.0]])  # one-pole lowpass
    low = sosfilt(sos, x.astype(np.float64)).astype(np.float32)
    t = float(rng.uniform(0.0, 0.4))
    if rng.random() < 0.5:
        return ((1 - t) * x + t * low).astype(np.float32)          # darker
    return ((1 + t) * x - t * low).astype(np.float32)              # brighter


def maybe_clip(x: np.ndarray, rng, prob: float) -> tuple[np.ndarray, float]:
    """With probability `prob`, drive the signal into hard clipping. Returns
    (clipped, equivalent_linear_gain) so callers can keep the ground truth's
    level aligned with the mixture's unclipped regions."""
    if rng.random() >= prob:
        return x, 1.0
    peak = float(np.max(np.abs(x)))
    if peak < EPS:
        return x, 1.0
    g = 0.99 / peak * float(rng.uniform(1.5, 3.0))
    return np.clip(x * g, -1.0, 1.0).astype(np.float32), g
