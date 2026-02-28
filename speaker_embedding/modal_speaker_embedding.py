"""
Modal Serverless GPU Inference Endpoint — Speaker Embedding Service
===================================================================

Deploys a SpeechBrain ECAPA-TDNN speaker verification model on Modal's
serverless GPU infrastructure.  Accepts a mono .wav file via HTTP POST
and returns a 192-dimensional speaker embedding vector as JSON.

Usage
-----
    modal deploy modal_speaker_embedding.py   # deploy to production
    modal serve  modal_speaker_embedding.py   # hot-reload dev server

Test
----
    curl -X POST https://<your-modal-url>/embed  \
         -H "Content-Type: audio/wav"             \
         --data-binary @sample.wav
"""

from __future__ import annotations

import io
import logging
from typing import Any

import modal

# ──────────────────────────────────────────────────────────────────────
# 1.  Modal App & Container Image
# ──────────────────────────────────────────────────────────────────────

app = modal.App("speaker-embedding-service")

# Debian-slim image with PyTorch (CUDA 12.1), torchaudio, and SpeechBrain.
# Modal uses uv internally for lightning-fast pip installs.
speaker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0,<2.5",
        "torchaudio>=2.2.0,<2.5",
        "speechbrain>=1.0.0,<1.1",
        "huggingface_hub>=0.24,<1.0",
        "requests",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Pre-install fastapi so the ASGI web layer is available at runtime.
    .pip_install("fastapi[standard]")
)


# ──────────────────────────────────────────────────────────────────────
# 2.  GPU-Backed Model Class  (@app.cls + @modal.enter)
# ──────────────────────────────────────────────────────────────────────


@app.cls(
    image=speaker_image,
    gpu="T4",                   # Cost-effective; swap to "A10G" for more VRAM
    timeout=300,                # 5-minute max per request
    container_idle_timeout=120, # Keep container warm for 2 min between calls
    allow_concurrent_inputs=4,  # Serve parallel requests on the same GPU
)
class SpeakerEmbeddingModel:
    """
    Wraps the ECAPA-TDNN speaker encoder.

    The model is downloaded from HuggingFace **once** on container boot
    via ``@modal.enter()``, then stays hot in GPU memory for every
    subsequent request — no cold-download penalty per call.
    """

    @modal.enter()
    def load_model(self) -> None:
        """Download and warm the ECAPA-TDNN model into GPU memory."""
        import torch
        import torchaudio

        # Compatibility shim: SpeechBrain 1.0.x calls
        # torchaudio.list_audio_backends() which may not exist.
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        from speechbrain.inference.speaker import EncoderClassifier

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("speaker-embedding")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info("Loading ECAPA-TDNN model on %s …", self.device)

        # Cache the model in /cache so it persists across container restarts
        # within the same deployment.
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/cache/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )
        self.logger.info("✓ Model loaded and ready for inference.")

    @modal.method()
    def compute_embedding(self, audio_bytes: bytes) -> list[float]:
        """
        Accept raw ``.wav`` bytes → resample → encode → return embedding.

        Parameters
        ----------
        audio_bytes : bytes
            Raw contents of a mono ``.wav`` file (3–5 s recommended).

        Returns
        -------
        list[float]
            192-dimensional speaker embedding.
        """
        import wave
        import numpy as np
        import torch
        import torchaudio

        # ── 1. Decode WAV (stdlib — avoids torchcodec/FFmpeg) ────────
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, T)
        self.logger.info(
            "Audio received: %d ch, %d Hz, %.2f s",
            n_channels,
            sample_rate,
            waveform.shape[1] / sample_rate,
        )

        # ── 2. (Mono downmix already handled above) ──────────────────

        # ── 3. Resample to 16 kHz (ECAPA-TDNN requirement) ──────────
        TARGET_SR = 16_000
        if sample_rate != TARGET_SR:
            self.logger.info("Resampling %d Hz → %d Hz", sample_rate, TARGET_SR)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=TARGET_SR
            ).to(self.device)
            waveform = resampler(waveform.to(self.device))
        else:
            waveform = waveform.to(self.device)

        # ── 4. Encode → 192-dim embedding ───────────────────────────
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
            # Shape: (1, 1, 192) → flatten to a plain Python list
            embedding_flat: list[float] = embedding.squeeze().cpu().tolist()

        self.logger.info("✓ Embedding computed — %d dimensions", len(embedding_flat))
        return embedding_flat


# ──────────────────────────────────────────────────────────────────────
# 3.  HTTP Endpoint  (FastAPI ASGI App)
# ──────────────────────────────────────────────────────────────────────
#
# We use @modal.asgi_app() with a full FastAPI instance so we can
# accept *raw binary bodies* (the .wav bytes) in the POST request.
# This is the production-recommended pattern for Modal web services.
# ──────────────────────────────────────────────────────────────────────


@app.function(image=speaker_image, timeout=120)
@modal.asgi_app()
def web_app():
    """Return a FastAPI application that exposes the /embed endpoint."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    api = FastAPI(
        title="Speaker Embedding Service",
        description=(
            "Upload a mono .wav file (3–5 s) and receive a "
            "192-dimensional ECAPA-TDNN speaker embedding."
        ),
        version="1.0.0",
    )

    # ── POST /embed ─────────────────────────────────────────────────
    @api.post("/embed")
    async def embed(request: Request) -> dict[str, Any]:
        """
        Accepts raw ``.wav`` bytes and returns the speaker embedding.

        **Request**

        - ``Content-Type: audio/wav`` (recommended)
        - Body: raw bytes of a mono .wav file

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

            # Dispatch to the GPU‐backed model container
            model = SpeakerEmbeddingModel()
            embedding: list[float] = model.compute_embedding.remote(audio_bytes)

            return {
                "embedding": embedding,
                "dimensions": len(embedding),
                "status": "ok",
            }

        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "detail": str(exc)},
            )

    # ── GET /health ─────────────────────────────────────────────────
    @api.get("/health")
    async def health() -> dict[str, str]:
        """Lightweight health-check / readiness probe."""
        return {"status": "ok"}

    return api
