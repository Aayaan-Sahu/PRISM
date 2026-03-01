"""
ECAPA-TDNN Enrollment Helper
============================

Given a path to a .wav file (from Dolphin or any clean speech source),
computes a 192-dim speaker embedding using the SpeechBrain ECAPA-TDNN model
and saves it to  noise_gate/embeddings/embeddingN.npy  where N is the next
available integer (never overwrites an existing file).

Importable API
--------------
    from ecapa_enroll import enroll_from_wav

    emb_path = enroll_from_wav("/path/to/speaker1_est.wav")
    print(f"Saved embedding → {emb_path}")

CLI
---
    uv run python av-tse/ecapa_enroll.py /path/to/speaker.wav
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torchaudio

# ── Paths ────────────────────────────────────────────────────────────────────

# Root of the spatial-audio-recognition repo (one level up from av-tse/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Where pretrained models are cached
_PRETRAINED_DIR = os.path.join(_REPO_ROOT, "pretrained_models", "spkrec-ecapa-voxceleb")

# Where embeddings are saved
_EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "noise_gate", "embeddings")

_MODEL_SR = 16_000  # ECAPA-TDNN expects 16 kHz mono


# ── Private: model singleton ─────────────────────────────────────────────────

_ecapa_model = None
_ecapa_device: str | None = None


def _load_model(device: str):
    """Load ECAPA-TDNN once and cache it."""
    global _ecapa_model, _ecapa_device

    if _ecapa_model is not None and _ecapa_device == device:
        return _ecapa_model

    from speechbrain.inference.speaker import EncoderClassifier

    print("[ECAPA] Loading ECAPA-TDNN model …", flush=True)
    _ecapa_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=_PRETRAINED_DIR,
        run_opts={"device": device},
    )
    _ecapa_device = device
    print(f"[ECAPA] Model loaded on {device}", flush=True)
    return _ecapa_model


# ── Private: next free embedding index ───────────────────────────────────────

def _next_embedding_path() -> str:
    """Return the path for the next free embeddingN.npy (no overwrites)."""
    os.makedirs(_EMBEDDINGS_DIR, exist_ok=True)
    n = 0
    while True:
        candidate = os.path.join(_EMBEDDINGS_DIR, f"embedding{n}.npy")
        if not os.path.exists(candidate):
            return candidate
        n += 1


# ── Public API ────────────────────────────────────────────────────────────────

def enroll_from_wav(wav_path: str, device: str | None = None) -> str:
    """
    Compute an ECAPA-TDNN embedding from a .wav file and save it.

    Parameters
    ----------
    wav_path : str
        Path to the source audio (any sample rate / channel count).
        Will be converted to 16 kHz mono internally.
    device : str | None
        PyTorch device string (e.g. 'cpu', 'mps', 'cuda').
        Defaults to 'mps' on Apple Silicon, 'cuda' if available, else 'cpu'.

    Returns
    -------
    str
        Absolute path of the saved .npy embedding file.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # ── Load audio ────────────────────────────────────────────────────────────
    print(f"[ECAPA] Loading audio: {wav_path}", flush=True)
    waveform, sr = torchaudio.load(wav_path)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz if needed
    if sr != _MODEL_SR:
        print(f"[ECAPA] Resampling {sr} Hz → {_MODEL_SR} Hz …", flush=True)
        waveform = torchaudio.transforms.Resample(sr, _MODEL_SR)(waveform)

    waveform = waveform.to(device)

    # ── Compute embedding ─────────────────────────────────────────────────────
    model = _load_model(device)
    with torch.no_grad():
        emb = model.encode_batch(waveform)   # (1, 1, 192)
        vec = emb.squeeze().cpu().numpy()    # (192,)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = _next_embedding_path()
    np.save(out_path, vec)

    print(f"[ECAPA] ✓ Saved {vec.shape[0]}-dim embedding → {out_path}", flush=True)
    print(f"[ECAPA]   L2 norm: {np.linalg.norm(vec):.4f}", flush=True)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ecapa_enroll.py <path/to/audio.wav>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    saved = enroll_from_wav(path)
    print(f"\nDone! Embedding saved to: {saved}")
