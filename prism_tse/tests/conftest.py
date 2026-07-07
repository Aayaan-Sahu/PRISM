"""Shared fixtures: a tiny synthetic corpus (fake 'speech', noise, RIRs and
manifests) so mixer/train tests run offline with no downloads."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from prism_tse.config import Config
from prism_tse.datasets.manifest import save_jsonl

SR = 16_000


def _fake_speech(rng: np.random.Generator, dur_s: float) -> np.ndarray:
    """Amplitude-modulated filtered noise — spectrally speech-ish enough for
    mechanical tests (nonzero, band-limited, with pauses)."""
    n = int(dur_s * SR)
    x = rng.standard_normal(n).astype(np.float32)
    # crude lowpass via cumulative smoothing
    kernel = np.hanning(33).astype(np.float32)
    x = np.convolve(x, kernel / kernel.sum(), mode="same")
    # syllable-rate envelope with silent gaps
    t = np.arange(n) / SR
    env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(2, 4) * t)).astype(np.float32)
    env *= (rng.random(n) < 0.995).astype(np.float32).cumprod() * 0 + 1  # keep simple
    x *= env
    peak = np.abs(x).max()
    return (x / peak * 0.5).astype(np.float32) if peak > 0 else x


def make_synth_corpus(root, n_speakers=6, utts_per_spk=3, n_val_speakers=2) -> Config:
    root = root if not isinstance(root, str) else __import__("pathlib").Path(root)
    rng = np.random.default_rng(0)
    wav_dir = root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = [], []
    for s in range(n_speakers + n_val_speakers):
        for u in range(utts_per_spk):
            dur = float(rng.uniform(4.0, 6.0))
            path = wav_dir / f"spk{s}_utt{u}.wav"
            sf.write(str(path), _fake_speech(rng, dur), SR)
            row = {"path": str(path), "speaker_id": f"spk{s}",
                   "duration_s": round(dur, 3), "sr": SR}
            (val_rows if s >= n_speakers else train_rows).append(row)

    noise_rows = []
    for i in range(4):
        dur = float(rng.uniform(6.0, 10.0))
        path = wav_dir / f"noise{i}.wav"
        sf.write(str(path), (rng.standard_normal(int(dur * SR)) * 0.1).astype(np.float32), SR)
        noise_rows.append({"path": str(path), "duration_s": round(dur, 3), "sr": SR})

    rir_rows = []
    for r in range(3):
        for src in range(2):
            n = int(0.3 * SR)
            rir = rng.standard_normal(n).astype(np.float32)
            rir *= np.exp(-np.arange(n) / (0.05 * SR)).astype(np.float32)
            rir[0] = 1.0  # direct path at t=0
            path = wav_dir / f"rir_room{r}_src{src}.wav"
            sf.write(str(path), rir / np.abs(rir).max(), SR)
            rir_rows.append({"room_id": f"room{r}", "path": str(path), "rt60": 0.3})

    mdir = root / "manifests"
    save_jsonl(mdir / "train_speech.jsonl", train_rows)
    save_jsonl(mdir / "val_speech.jsonl", val_rows)
    save_jsonl(mdir / "noise_train.jsonl", noise_rows)
    save_jsonl(mdir / "noise_val.jsonl", noise_rows)
    save_jsonl(mdir / "rirs.jsonl", rir_rows)

    cfg = Config()
    cfg.paths.data_root = str(root)
    return cfg


@pytest.fixture(scope="session")
def synth_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_corpus")
    return make_synth_corpus(root)
