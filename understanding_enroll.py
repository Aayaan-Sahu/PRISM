import os
import sys
import numpy as np
import torch
import torchaudio

_ecapa_model = None
_ecapa_device: str | None = None

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
_PRETRAINED_DIR = os.path.join(_REPO_ROOT, "pretrained_models", "spkrec-ecapa-voxceleb")
_EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "embeddings")
_MODEL_SR = 16_000


def _next_embedding_path() -> str:
    os.makedirs(_EMBEDDINGS_DIR, exist_ok=True)
    n = 0
    while True:
        candidate = os.path.join(_EMBEDDINGS_DIR, f"embedding{n}.npy")
        if not os.path.exists(candidate):
            return candidate
        n += 1


def _load_model(device: str):
    global _ecapa_model, _ecapa_device

    if _ecapa_model is not None and _ecapa_device == device:
        return _ecapa_model

    from speechbrain.inference.speaker import EncoderClassifier

    print("[ECAPA] Loading ECAPA-TDNN model...")
    _ecapa_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=_PRETRAINED_DIR,
        run_opts={"device": device},
    )
    _ecapa_device = device
    print(f"[ECAPA] Model loaded on {device}")
    return _ecapa_model


def enroll_from_wav(wav_path: str, device: str | None = None) -> str:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[ECAPA] Loading audio: {wav_path}")
    waveform, sr = torchaudio.load(wav_path)

    # ensure mono audio
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # ensure audio is right sample rate
    if sr != _MODEL_SR:
        print(f"[ECAPA] Resampling {sr} Hz -> {_MODEL_SR} Hz...")
        waveform = torchaudio.transforms.Resample(sr, _MODEL_SR)(waveform)

    waveform = waveform.to(device)

    # compute embeddings
    model = _load_model(device)
    with torch.no_grad():
        emb = model.encode_batch(waveform)
        vec = emb.squeeze()

    # compute l2 normalization to enable cosine similarity later
    norm = torch.norm(vec, p=2)
    if norm.item() < 1e-8:
        raise ValueError(f"Computed embedding for {wav_path} has near-zero norm")
    vec = (vec / norm).cpu().numpy()

    # save the embedding (fingerprint) to next available embedding
    out_path = _next_embedding_path()
    np.save(out_path, vec)

    # save the reference waveform for WeSep target speaker extraction
    ref_path = out_path.replace(".npy", "_ref.npy")
    np.save(ref_path, waveform.squeeze().cpu().numpy())

    print(f"[ECAPA] Saved {vec.shape[0]}-dim embedding -> {out_path}")
    print(f"[ECAPA] Saved reference waveform -> {ref_path}")
    return out_path


# ---------------------------------------------------------------------------
# Modal Volume upload — call this after enroll_from_wav() to push files to
# the persistent 'tse-embeddings' Volume so the running TSE container can
# see the new speaker without a re-deploy.
#
# Uses Modal's local batch_upload() API — writes directly from this machine
# to the Volume, no Modal function needed.
# ---------------------------------------------------------------------------
import io
import modal as _modal

_embeddings_volume = _modal.Volume.from_name("tse-embeddings", create_if_missing=True)


def upload_embedding(
    emb_filename: str, emb_data: bytes,
    ref_filename: str, ref_data: bytes,
):
    """
    Write a speaker's embedding + reference waveform into the Modal Volume.
    Both files must be uploaded together because TSERuntime.load_embeddings()
    skips any speaker missing the _ref.npy file.
    """
    with _embeddings_volume.batch_upload() as batch:
        batch.put_file(io.BytesIO(emb_data), emb_filename)
        batch.put_file(io.BytesIO(ref_data), ref_filename)
    print(f"[UPLOAD] Wrote {emb_filename} + {ref_filename} to Volume")


