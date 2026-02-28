"""
Modal Cloud Separator — GPU-Accelerated Source Separation & Biometric Gating
=============================================================================

Deploys SepFormer (speech separation) and ECAPA-TDNN (speaker verification)
on an NVIDIA H100 via Modal.  Exposes a FastAPI WebSocket that:

  1. Receives enrolled speaker embeddings (N×192) on connect.
  2. Streams 150 ms int16 PCM chunks from the client.
  3. Runs overlap-save SepFormer separation on a 1 s sliding buffer.
  4. Checks each separated track against enrolled embeddings (cosine sim).
  5. Mixes result: enrolled tracks at 1.0× gain, rejected tracks at 0.05×.
  6. Streams 150 ms int16 PCM back to the client.

Deploy
------
    modal deploy modal_separator.py

Local dev
---------
    modal serve modal_separator.py
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Modal image — install all deps and cache both models at build time
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "speechbrain>=1.0.3",
        "fastapi[standard]",
        "uvicorn",
        "numpy>=2.0",
        "huggingface_hub<1.0",
        "requests",
    )
    .run_commands(
        # Pre-download both models into the container so cold starts are fast
        "python -c \""
        "from speechbrain.inference.separation import SepformerSeparation; "
        "SepformerSeparation.from_hparams("
        "  source='speechbrain/sepformer-whamr16k',"
        "  savedir='/models/sepformer-whamr16k'"
        ")\"",
        "python -c \""
        "from speechbrain.inference.speaker import EncoderClassifier; "
        "EncoderClassifier.from_hparams("
        "  source='speechbrain/spkrec-ecapa-voxceleb',"
        "  savedir='/models/spkrec-ecapa-voxceleb'"
        ")\"",
    )
)

app = modal.App("spatial-audio-separator", image=image)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SR = 16_000                               # SepFormer & ECAPA sample rate
CHUNK_DURATION = 0.15                            # 150 ms chunks from client
CHUNK_SAMPLES = int(MODEL_SR * CHUNK_DURATION)   # 2400 samples
CONTEXT_DURATION = 1.0                           # 1 s sliding window for SepFormer
CONTEXT_SAMPLES = int(MODEL_SR * CONTEXT_DURATION)  # 16000 samples

SIMILARITY_THRESHOLD = 0.10                      # Match the local VAD threshold
REJECT_GAIN = 0.0                               # −26 dB dim for non-target speakers

# ---------------------------------------------------------------------------
# Embedded FastAPI app source — written to disk at container startup.
# Using @modal.web_server to bypass Modal's built-in ASGI adapter, which
# has a known TypeError on WebSocket connections.  Uvicorn serves our
# FastAPI app directly and handles WebSocket upgrades natively.
# ---------------------------------------------------------------------------

FASTAPI_APP_CODE = r'''
"""Spatial Audio Separator — in-container FastAPI app (served by uvicorn)."""

import json
import struct
import sys
import time
import traceback

import numpy as np
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

MODEL_SR = 16_000
CHUNK_SAMPLES = int(MODEL_SR * 0.15)
CONTEXT_SAMPLES = int(MODEL_SR * 1.0)
SIMILARITY_THRESHOLD = 0.35
HANG_THRESHOLD = 0.25      # Lower threshold to KEEP gate open
HANG_TIME = 0.8            # Seconds to keep gate open after Target stops
REJECT_GAIN = 0.00

# ── Model state ──────────────────────────────────────────────────────
sep_model = None
ecapa_model = None
device = None
load_error = None

def load_models():
    global sep_model, ecapa_model, device, load_error
    try:
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.inference.speaker import EncoderClassifier

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

        print("Loading SepFormer ...", flush=True)
        sep_model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir="/models/sepformer-whamr16k",
            run_opts={"device": str(device)},
        )
        print("SepFormer loaded", flush=True)

        print("Loading ECAPA-TDNN ...", flush=True)
        ecapa_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/models/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)},
        )
        print("ECAPA-TDNN loaded", flush=True)
    except Exception as e:
        load_error = traceback.format_exc()
        print(f"MODEL LOAD FAILED: {load_error}", file=sys.stderr, flush=True)

load_models()

# ── FastAPI app ──────────────────────────────────────────────────────
web_app = FastAPI(title="Spatial Audio Separator")

@web_app.get("/health")
async def health():
    return {
        "status": "ok" if sep_model is not None else "models_not_loaded",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "error": load_error,
    }

@web_app.post("/enroll")
async def enroll(request: Request):
    """Accept raw int16 PCM audio, run SepFormer -> ECAPA, return embedding."""
    if sep_model is None or ecapa_model is None:
        return JSONResponse({"error": f"Models not loaded: {load_error}"}, status_code=503)

    raw = await request.body()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0).to(device)
    print(f"Enroll: received {len(samples)} samples ({len(samples)/MODEL_SR:.1f}s)", flush=True)

    with torch.no_grad():
        # Run through SepFormer — take the dominant (louder) track
        est_sources = sep_model.separate_batch(waveform)
        src1 = est_sources[0, :, 0]
        src2 = est_sources[0, :, 1]
        dominant = src1 if src1.abs().mean() > src2.abs().mean() else src2

        # Compute ECAPA embedding on the SepFormer-processed audio
        embedding = ecapa_model.encode_batch(dominant.unsqueeze(0))
        vec = embedding.squeeze().cpu().numpy().tolist()

    print(f"Enroll: returning {len(vec)}-dim embedding", flush=True)
    return JSONResponse({"embedding": vec})

@web_app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected", flush=True)

    if sep_model is None:
        await websocket.send_text(json.dumps({
            "type": "error", "detail": f"Models not loaded: {load_error}"
        }))
        await websocket.close()
        return

    # 1. Handshake
    try:
        init_raw = await websocket.receive_text()
        init_msg = json.loads(init_raw)
        assert init_msg["type"] == "init"

        emb_list = init_msg["embeddings"]
        n_speakers = len(emb_list)
        target_emb = torch.tensor(emb_list, dtype=torch.float32, device=device)
        print(f"Loaded {n_speakers} target embedding(s)", flush=True)

        await websocket.send_text(json.dumps({
            "type": "ready",
            "speakers": n_speakers,
            "chunk_samples": CHUNK_SAMPLES,
        }))
    except Exception as e:
        print(f"Handshake error: {e}", flush=True)
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({"type": "error", "detail": str(e)}))
            await websocket.close()
        except Exception:
            pass
        return

    # 2. Streaming loop
    context_buf = torch.zeros(1, CONTEXT_SAMPLES, dtype=torch.float32, device=device)
    chunks_received = 0
    
    # Hysteresis state (per track)
    track_is_target = [False, False]
    track_last_time = [0.0, 0.0]

    try:
        while True:
            raw = await websocket.receive_bytes()
            t0 = time.perf_counter()

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            chunk_tensor = torch.from_numpy(samples).to(device).unsqueeze(0)
            chunk_len = chunk_tensor.shape[1]

            context_buf = torch.cat([context_buf[:, chunk_len:], chunk_tensor], dim=1)
            chunks_received += 1

            if chunks_received < 4:
                silence = np.zeros(chunk_len, dtype=np.int16)
                header = struct.pack('ffff', 0.0, 0.0, 0.0, 0.0)
                await websocket.send_bytes(header + silence.tobytes())
                continue

            with torch.no_grad():
                try:
                    if chunks_received == 4:
                        print(f"FIRST SEPARATION: context_buf shape={context_buf.shape}, dtype={context_buf.dtype}", flush=True)

                    est_sources = sep_model.separate_batch(context_buf)

                    if chunks_received == 4:
                        print(f"SEPARATION OK: est_sources shape={est_sources.shape}", flush=True)

                    src1 = est_sources[0, -chunk_len:, 0]
                    src2 = est_sources[0, -chunk_len:, 1]

                    # Use FULL 1s separated tracks for ECAPA (not just 150ms)
                    src1_full = est_sources[0, :, 0]  # (16000,)
                    src2_full = est_sources[0, :, 1]  # (16000,)
                    sources_for_ecapa = torch.stack([src1_full, src2_full], dim=0)

                    src_embeddings = ecapa_model.encode_batch(sources_for_ecapa).squeeze(1)

                    sims = torch.nn.functional.cosine_similarity(
                        src_embeddings.unsqueeze(1), target_emb.unsqueeze(0), dim=-1
                    )
                    max_sims, _ = sims.max(dim=1)

                    t_now = time.time()
                    gains = torch.full_like(max_sims, REJECT_GAIN)

                    for i in range(2):
                        sim_i = max_sims[i].item()
                        if sim_i > SIMILARITY_THRESHOLD:
                            track_is_target[i] = True
                            track_last_time[i] = t_now
                        elif track_is_target[i] and sim_i > HANG_THRESHOLD:
                            track_is_target[i] = True
                            track_last_time[i] = t_now
                        else:
                            if t_now - track_last_time[i] < HANG_TIME:
                                track_is_target[i] = True
                            else:
                                track_is_target[i] = False

                        if track_is_target[i]:
                            gains[i] = 1.0

                    output = gains[0] * src1 + gains[1] * src2
                    output = torch.clamp(output, -1.0, 1.0)
                    out_int16 = (output * 32767.0).to(torch.int16).cpu().numpy()
                except Exception as sep_err:
                    print(f"SEPARATION ERROR on chunk {chunks_received}: {type(sep_err).__name__}: {sep_err}", flush=True)
                    traceback.print_exc()
                    # Fallback: echo the raw chunk back
                    out_int16 = np.frombuffer(raw, dtype=np.int16)
                    max_sims = torch.zeros(2)
                    gains = torch.ones(2)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            if chunks_received % 10 == 0:
                print(
                    f"  chunk {chunks_received:4d} | "
                    f"sims=[{max_sims[0]:.2f}, {max_sims[1]:.2f}] "
                    f"gains=[{gains[0]:.2f}, {gains[1]:.2f}] | "
                    f"{elapsed_ms:.1f}ms",
                    flush=True,
                )

            header = struct.pack('ffff',
                float(max_sims[0]), float(max_sims[1]),
                float(gains[0]), float(gains[1]))
            await websocket.send_bytes(header + out_int16.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected", flush=True)
    except Exception as e:
        print(f"Streaming error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    finally:
        print("Connection cleanup complete", flush=True)
'''


# ---------------------------------------------------------------------------
# Modal web_server entrypoint — runs uvicorn directly, bypassing Modal's
# ASGI adapter to get native WebSocket support.
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    scaledown_window=300,
    image=image,
)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    """Write the FastAPI app to disk and launch uvicorn."""
    import os
    import subprocess

    # Ensure the /app directory exists
    os.makedirs("/app", exist_ok=True)

    # Write the embedded app code to a file
    with open("/app/server.py", "w") as f:
        f.write(FASTAPI_APP_CODE)

    # Launch uvicorn — Modal will route traffic to port 8000
    subprocess.Popen(
        [
            "python", "-m", "uvicorn",
            "server:web_app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info",
        ],
        cwd="/app",
    )


# ---------------------------------------------------------------------------
# CLI entry point for `modal run modal_separator.py`
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("✓ Modal app 'spatial-audio-separator' is defined correctly.")
    print("  To deploy:   modal deploy modal_separator.py")
    print("  To dev mode: modal serve modal_separator.py")
