"""
Local Speaker Embedding Server (no Modal required)
===================================================

Drop-in local replacement for the Modal GPU endpoint.
Runs a FastAPI/Uvicorn server on localhost so you can test the
embedding pipeline without deploying anything.

Usage
-----
    # Install deps first (one-time)
    pip install torch torchaudio speechbrain fastapi uvicorn

    # Start the server
    python local_speaker_embedding.py

    # Test
    curl -X POST http://localhost:8000/embed  \
         -H "Content-Type: audio/wav"          \
         --data-binary @sample.wav
"""

from __future__ import annotations

import io
import logging
from typing import Any

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from speechbrain.inference.speaker import EncoderClassifier

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("speaker-embedding")

# ──────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────

api = FastAPI(
    title="Speaker Embedding Service (Local)",
    description=(
        "Upload a mono .wav file (3–5 s) and receive a "
        "192-dimensional ECAPA-TDNN speaker embedding."
    ),
    version="1.0.0-local",
)

# ──────────────────────────────────────────────────────────────────────
# Model Loading (runs once at startup)
# ──────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: EncoderClassifier | None = None


@api.on_event("startup")
async def load_model() -> None:
    """Download ECAPA-TDNN on first launch; cached to ./pretrained_models."""
    global model
    logger.info("Loading ECAPA-TDNN model on %s …", device)
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    logger.info("✓ Model loaded and ready for inference.")


# ──────────────────────────────────────────────────────────────────────
# Embedding Helper
# ──────────────────────────────────────────────────────────────────────


def compute_embedding(audio_bytes: bytes) -> list[float]:
    """
    Decode .wav bytes → resample to 16 kHz → ECAPA-TDNN → 192-dim list.
    """
    assert model is not None, "Model not loaded yet."

    # 1. Decode WAV
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    logger.info(
        "Audio received: %d ch, %d Hz, %.2f s",
        waveform.shape[0],
        sample_rate,
        waveform.shape[1] / sample_rate,
    )

    # 2. Downmix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3. Resample to 16 kHz
    TARGET_SR = 16_000
    if sample_rate != TARGET_SR:
        logger.info("Resampling %d Hz → %d Hz", sample_rate, TARGET_SR)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SR
        ).to(device)
        waveform = resampler(waveform.to(device))
    else:
        waveform = waveform.to(device)

    # 4. Encode → 192-dim embedding
    with torch.no_grad():
        embedding = model.encode_batch(waveform)
        embedding_flat: list[float] = embedding.squeeze().cpu().tolist()

    logger.info("✓ Embedding computed — %d dimensions", len(embedding_flat))
    return embedding_flat


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@api.post("/embed")
async def embed(request: Request) -> dict[str, Any]:
    """
    POST /embed — send raw .wav bytes, receive a 192-dim speaker embedding.

    **Request**
    - ``Content-Type: audio/wav`` (recommended)
    - Body: raw bytes of a mono .wav file (3–5 s)

    **Response 200**
    ```json
    {
        "embedding": [0.012, -0.045, ...],
        "dimensions": 192,
        "status": "ok"
    }
    ```
    """
    try:
        audio_bytes: bytes = await request.body()

        if not audio_bytes:
            return JSONResponse(
                status_code=422,
                content={
                    "status": "error",
                    "detail": "Empty request body — expected .wav bytes.",
                },
            )

        embedding = compute_embedding(audio_bytes)

        return {
            "embedding": embedding,
            "dimensions": len(embedding),
            "status": "ok",
        }

    except Exception as exc:
        logger.exception("Embedding failed")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(exc)},
        )


@api.get("/health")
async def health() -> dict[str, str]:
    """Lightweight readiness check."""
    return {"status": "ok", "device": str(device)}


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "local_speaker_embedding:api",
        host="0.0.0.0",
        port=8000,
        reload=True,   # auto-reload on file changes during development
        log_level="info",
    )
