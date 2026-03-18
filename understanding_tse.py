"""
Modal Cloud TSE (WeSep) — Understanding Version
=================================================

How this works (two-step pipeline):

1. WeSep (Target Speaker Extraction)
   - NOT blind source separation — it doesn't split into N tracks
   - Takes: mixed audio + reference waveform of a target speaker
   - Returns: only that speaker's voice, suppressing everyone else
   - If the speaker is absent, it returns noise/artifacts (not silence)

2. ECAPA-TDNN (Verification Gate)
   - Embeds WeSep's output and compares against the stored enrollment embedding
   - If cosine similarity >= threshold → pass the separated audio through
   - If below threshold → WeSep output was noise/leakage → return silence

This ensures that if two people are talking and only one is enrolled,
only the enrolled person's audio is returned.

Deploy:  modal deploy understanding_tse.py
Dev:     modal serve  understanding_tse.py
"""

import collections
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import modal

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_WESEP_LOCAL_DIR = os.path.join(_REPO_ROOT, "third_party", "wesep")
_WESPEAKER_LOCAL_DIR = os.path.join(_REPO_ROOT, "third_party", "wespeaker")
_EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "embeddings")

# ---------------------------------------------------------------------------
# Modal image — container environment
# ---------------------------------------------------------------------------
tse_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "speechbrain>=1.0.3",
        "fastapi[standard]",
        "uvicorn",
        "numpy>=2.0",
        "soundfile",
        "pyyaml",
        "silero-vad",
        "kaldiio",
        "requests",
        "huggingface_hub<1.0",
    )
    .add_local_dir(_WESEP_LOCAL_DIR, remote_path="/app/third_party/wesep", copy=True)
    .add_local_dir(_WESPEAKER_LOCAL_DIR, remote_path="/app/third_party/wespeaker", copy=True)
    # NOTE: embeddings are NOT baked into the image — they live in a Modal Volume
    # so newly enrolled speakers are available without re-deploying.
    # Pre-download models at image build time so container startup is fast
    .run_commands(
        "python -c \""
        "from speechbrain.inference.speaker import EncoderClassifier; "
        "EncoderClassifier.from_hparams("
        "  source='speechbrain/spkrec-ecapa-voxceleb',"
        "  savedir='/models/spkrec-ecapa-voxceleb'"
        ")"
        "\"",
        "python -c \""
        "import sys; "
        "sys.path.insert(0, '/app/third_party/wespeaker'); "
        "sys.path.insert(0, '/app/third_party/wesep'); "
        "from wesep.cli.hub import Hub; "
        "print('WeSep model dir:', Hub.get_model('english'))"
        "\"",
    )
)

app = modal.App("spatial-audio-tse", image=tse_image)

# Persistent volume for enrolled speaker data (embedding + reference waveform pairs).
# Using a Volume instead of image copy means new enrollments are immediately visible
# to the running container after a reload_embeddings event — no re-deploy needed.
embeddings_volume = modal.Volume.from_name("tse-embeddings", create_if_missing=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_SAMPLE_RATE = 16_000
CHUNK_SAMPLES = int(MODEL_SAMPLE_RATE * 0.15)       # 150 ms per chunk from client
WINDOW_SAMPLES = int(MODEL_SAMPLE_RATE * 2.0)       # 2 s sliding window for WeSep context
SIMILARITY_THRESHOLD = 0.35                          # cosine sim gate for ECAPA verification
WARMUP_CHUNKS = 4                                    # fill the window before processing
EMBEDDINGS_PATH = "/app/embeddings"                  # where enrolled data lives in-container


def _ensure_paths():
    """Add third-party dirs to sys.path so we can import wesep/wespeaker."""
    for p in ("/app/third_party/wespeaker", "/app/third_party/wesep"):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Modal class — lifecycle-managed model loading + ASGI app
# ---------------------------------------------------------------------------
@app.cls(
    gpu="H100",
    scaledown_window=300,
    image=tse_image,
    volumes={"/app/embeddings": embeddings_volume},  # mount the live Volume here
)
class TSERuntime:

    @modal.enter()
    def load_models(self):
        """Called once when a new container spins up. Loads both models to GPU."""
        self.extractor = None
        self.tdnn = None
        self.device = None

        try:
            _ensure_paths()

            import torch
            from wesep import load_model
            from speechbrain.inference.speaker import EncoderClassifier

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[TSE] device={self.device}", flush=True)
            if torch.cuda.is_available():
                print(f"[TSE] gpu={torch.cuda.get_device_name(0)}", flush=True)

            # Load WeSep separation model
            self.extractor = load_model("english")
            self.extractor.set_device(str(self.device))
            self.extractor.set_resample_rate(MODEL_SAMPLE_RATE)
            # TODO: Should set_vad be false?
            self.extractor.set_vad(False)
            self.extractor.set_wavform_norm(True)
            self.extractor.set_output_norm(False)
            print("[TSE] WeSep extractor loaded", flush=True)

            # ECAPA-TDNN: speaker embedding model (verification gate)
            self.tdnn = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)},
            )
            print("[TSE] ECAPA-TDNN loaded", flush=True)

        except Exception:
            print(f"[TSE] MODEL LOAD FAILED:\n{traceback.format_exc()}", file=sys.stderr, flush=True)

    # -- Helper methods --

    def load_embeddings(self) -> list[dict]:
        """
        Scan /app/embeddings for pairs of embeddingN.npy + embeddingN_ref.npy.
        Returns a list of {"id", "emb", "ref_wav"} dicts ready for WeSep + ECAPA.
        """
        enrolled = []
        if not os.path.isdir(EMBEDDINGS_PATH):
            print(f"[TSE] No embeddings directory at {EMBEDDINGS_PATH}", flush=True)
            return enrolled

        # Find all embedding files (not _ref files)
        emb_files = sorted(
            f for f in os.listdir(EMBEDDINGS_PATH)
            if f.endswith(".npy") and "_ref" not in f
        )

        for emb_file in emb_files:
            ref_file = emb_file.replace(".npy", "_ref.npy")
            emb_path = os.path.join(EMBEDDINGS_PATH, emb_file)
            ref_path = os.path.join(EMBEDDINGS_PATH, ref_file)

            if not os.path.exists(ref_path):
                print(f"[TSE] Skipping {emb_file}: no matching {ref_file}", flush=True)
                continue

            try:
                # Load embedding, L2-normalize, move to GPU
                emb_np = np.load(emb_path)
                emb_t = torch.from_numpy(emb_np).float().to(self.device)
                norm = emb_t.norm(p=2)
                if norm.item() > 1e-8:
                    emb_t = emb_t / norm

                # Load reference waveform as (1, T) tensor
                ref_np = np.load(ref_path)
                ref_wav = torch.from_numpy(ref_np).float().unsqueeze(0)

                speaker_id = emb_file.replace(".npy", "")
                enrolled.append({"id": speaker_id, "emb": emb_t, "ref_wav": ref_wav})
                print(f"[TSE] Loaded speaker: {speaker_id}", flush=True)

            except Exception as e:
                print(f"[TSE] Failed to load {emb_file}: {e}", flush=True)

        print(f"[TSE] {len(enrolled)} speaker(s) enrolled", flush=True)
        return enrolled

    def compute_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Run ECAPA-TDNN on a waveform and return an L2-normalized embedding.
        waveform: (1, T) float32 tensor (CPU is fine, we move to device).
        """
        wav = waveform.to(self.device).float()
        with torch.no_grad():
            emb = self.tdnn.encode_batch(wav).squeeze()
        if emb.ndim == 0:
            emb = emb.unsqueeze(0)
        norm = emb.norm(p=2)
        if norm.item() < 1e-8:
            return torch.zeros_like(emb)
        return emb / norm

    def extract_for_target(self, context: torch.Tensor, ref_wav: torch.Tensor) -> torch.Tensor:
        """
        Run WeSep to extract one speaker's voice from the context window.
        context:  (1, WINDOW_SAMPLES) mixed audio
        ref_wav:  (1, T) reference waveform of the target speaker
        Returns:  (1, WINDOW_SAMPLES) separated audio, padded/trimmed to match context length.
        """
        pred = self.extractor.extract_speech_from_pcm(
            context, MODEL_SAMPLE_RATE, ref_wav, MODEL_SAMPLE_RATE
        )
        if pred is None:
            return torch.zeros_like(context)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        if pred.shape[0] > 1:
            pred = pred[:1]

        # Pad or trim to match context length exactly
        target_len = context.shape[1]
        current_len = pred.shape[1]
        if current_len < target_len:
            pred = F.pad(pred, (0, target_len - current_len))
        elif current_len > target_len:
            pred = pred[:, :target_len]

        return pred.float().cpu()

    # -- ASGI app --

    @modal.asgi_app()
    def serve(self):
        web_app = FastAPI(title="WeSep TSE")

        @web_app.get("/health")
        async def health():
            return {
                "status": "ok" if self.extractor is not None and self.tdnn is not None else "models_not_loaded",
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            }

        @web_app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("[TSE] Client connected", flush=True)

            # Guard: models must be loaded
            if self.extractor is None or self.tdnn is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "detail": "Models not loaded",
                }))
                await websocket.close()
                return

            # -- Per-connection state --
            # Sliding window: deque auto-evicts old samples when full (same as enroll_demo.py)
            audio_buf = collections.deque(maxlen=WINDOW_SAMPLES)
            enrolled = self.load_embeddings()
            chunks_received = 0

            # Tell the client we're ready
            await websocket.send_text(json.dumps({
                "type": "ready",
                "speakers_loaded": len(enrolled),
                "sample_rate": MODEL_SAMPLE_RATE,
                "chunk_samples": CHUNK_SAMPLES,
                "window_samples": WINDOW_SAMPLES,
            }))

            try:
                while True:
                    msg = await websocket.receive()

                    # --- Text messages: commands (e.g. reload embeddings) ---
                    if "text" in msg and msg["text"]:
                        try:
                            payload = json.loads(msg["text"])
                            if payload.get("type") == "reload_embeddings":
                                # CRITICAL: reload the Volume so the container sees newly uploaded files.
                                # Without this, the container has a stale snapshot from mount time.
                                embeddings_volume.reload()
                                print(f"[TSE] Volume reloaded from disk", flush=True)
                                # Now scan for embedding files
                                enrolled = self.load_embeddings()
                                await websocket.send_text(json.dumps({
                                    "type": "embeddings_loaded",
                                    "count": len(enrolled),
                                }))
                                print(f"[TSE] Hot-reloaded: {len(enrolled)} speaker(s)", flush=True)
                        except Exception as e:
                            print(f"[TSE] Reload error: {e}", flush=True)
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "detail": f"reload failed: {e}",
                            }))
                        continue

                    # --- Binary messages: PCM16 audio chunks ---
                    raw = msg.get("bytes") if isinstance(msg, dict) else None
                    if not raw:
                        continue

                    t0 = time.perf_counter()

                    # Decode PCM16 → float32 [-1, 1]
                    chunk_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    chunk_len = len(chunk_np)

                    # Append to sliding window (old samples auto-evict)
                    audio_buf.extend(chunk_np)
                    chunks_received += 1

                    # Debug: log every 20th chunk so we can see audio is flowing
                    if chunks_received % 20 == 0:
                        energy = float(np.mean(chunk_np ** 2))
                        print(f"[TSE] chunk#{chunks_received} len={chunk_len} energy={energy:.6f} buf={len(audio_buf)}/{WINDOW_SAMPLES}", flush=True)

                    # Wait until the window is full before processing
                    if len(audio_buf) < WINDOW_SAMPLES:
                        if chunks_received % 20 == 0:
                            print(f"[TSE] Filling buffer... {len(audio_buf)}/{WINDOW_SAMPLES}", flush=True)
                        await websocket.send_bytes(np.zeros(chunk_len, dtype=np.int16).tobytes())
                        continue

                    # No enrolled speakers → silence
                    if not enrolled:
                        if chunks_received % 20 == 0:
                            print(f"[TSE] No speakers enrolled, returning silence", flush=True)
                        await websocket.send_bytes(np.zeros(chunk_len, dtype=np.int16).tobytes())
                        continue

                    # Build the 2s context tensor from the deque
                    window = torch.from_numpy(np.array(audio_buf)).unsqueeze(0).float()  # (1, WINDOW_SAMPLES)

                    # Accumulator for output — supports multiple enrolled speakers
                    mix_out = torch.zeros(chunk_len, dtype=torch.float32)
                    best_sim = 0.0

                    for speaker in enrolled:
                        try:
                            # Step 1: WeSep extracts the target speaker's voice
                            separated = self.extract_for_target(window, speaker["ref_wav"])

                            # Step 2: ECAPA embeds the separated output for verification
                            sep_emb = self.compute_embedding(separated)

                            # Step 3: Cosine similarity gate
                            sim = F.cosine_similarity(
                                sep_emb.unsqueeze(0),
                                speaker["emb"].unsqueeze(0),
                            ).item()

                            best_sim = max(best_sim, sim)

                            # Debug: log match/no-match for every 10th chunk
                            if chunks_received % 10 == 0:
                                status = "✅ MATCH" if sim >= SIMILARITY_THRESHOLD else "❌ no match"
                                sep_energy = float(separated[0, -chunk_len:].pow(2).mean())
                                print(f"[TSE] speaker={speaker['id']} sim={sim:.3f} {status} sep_energy={sep_energy:.6f}", flush=True)

                            if sim >= SIMILARITY_THRESHOLD:
                                # Only take the tail chunk_len samples (the new data)
                                mix_out += separated[0, -chunk_len:]

                        except Exception as e:
                            print(f"[TSE] speaker={speaker['id']} error={e}", flush=True)
                            traceback.print_exc()

                    # Clamp to [-1, 1] and convert back to PCM16
                    mix_out = torch.clamp(mix_out, -1.0, 1.0)
                    out_i16 = (mix_out * 32767.0).to(torch.int16).cpu().numpy()
                    await websocket.send_bytes(out_i16.tobytes())

                    # Periodic stats
                    if chunks_received % 10 == 0:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        await websocket.send_text(json.dumps({
                            "type": "stats",
                            "best_similarity": round(best_sim, 3),
                            "process_ms": round(elapsed_ms, 1),
                            "chunks": chunks_received,
                        }))

            except WebSocketDisconnect:
                print("[TSE] Client disconnected", flush=True)
            except Exception as e:
                print(f"[TSE] Streaming error: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                try:
                    await websocket.send_text(json.dumps({"type": "error", "detail": str(e)}))
                except Exception:
                    pass
            finally:
                print("[TSE] Connection cleanup complete", flush=True)

        return web_app


@app.local_entrypoint()
def main():
    print("Modal app 'spatial-audio-tse' is defined.")
    print("  deploy: modal deploy understanding_tse.py")
    print("  dev:    modal serve  understanding_tse.py")