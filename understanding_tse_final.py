from __future__ import annotations

import base64
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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_WESEP_LOCAL_DIR = os.path.join(_REPO_ROOT, "third_party", "wesep")
_WESPEAKER_LOCAL_DIR = os.path.join(_REPO_ROOT, "third_party", "wespeaker")

# Create an image to spin up modal container from
# Has all the required dependencies and inserts wespeaker and wesep so we can import
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
        "soundfile",
        "pyyaml",
        "silero-vad",
        "kaldiio",
        "requests",
        "huggingface_hub<1.0",
    )
    .add_local_dir(_WESEP_LOCAL_DIR, remote_path="/app/third_party/wesep", copy=True)
    .add_local_dir(_WESPEAKER_LOCAL_DIR, remote_path="/app/third_party/wespeaker", copy=True)
    # the encoderclassifier.from_hparams command does the following
    # find the speechbrain/spkrec-ecapa-voxceleb
    # download it to the remove models folder on the gpu (if it doesn't exist - downloads it)
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

app = modal.App("spatial-audio-tse", image=image)

# Domain Constants
MODEL_SR = 16000
CHUNK_SAMPLES = int(MODEL_SR * 0.15)
CONTEXT_SAMPLES = int(MODEL_SR * 1.0)
MAX_TARGETS = 8

SIMILARITY_THRESHOLD = 0.35
HANG_THRESHOLD = 0.25
HANG_TIME = 0.8
STATS_EVERY_N_CHUNKS = 10


def _ensure_paths():
    for p in ("/app/third_party/wespeaker", "/app/third_party/wesep"):
        if p not in sys.path:
            sys.path.insert(0, p)


@app.cls(gpu="H100", scaledown_window=300, image=image)
class TSERuntime:
    @modal.enter()
    def load_models(self):
        self.load_error = None
        self.extractor = None
        self.spk_model = None
        self.device = None

        try:
            _ensure_paths()

            from wesep import load_model
            from speechbrain.inference.speaker import EncoderClassifier

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[WESEP]: device={self.device}", flush=True)
            if torch.cuda.is_available():
                print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

            self.extractor = load_model("english")
            self.extractor.set_device(str(self.device))
            self.extractor.set_resample_rate(MODEL_SR)
            self.extractor.set_vad(False)
            self.extractor.set_wavform_norm(True)
            self.extractor.set_output_norm(False)
            print("WeSep extractor loaded", flush=True)

            # load the ecapa-tdnn model
            self.spk_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)},
            )
            print("ECAPA speaker model loaded", flush=True)

        except Exception:
            self.load_error = traceback.format_exc()
            print(f"MODEL LOAD FAILED:\n{self.load_error}", file=sys.stderr, flush=True)

    def _fit_to_length(self, wav: torch.Tensor, length: int) -> torch.Tensor:
        # forces mono audio
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[:1]

        current = wav.shape[1]
        # if curr length is too small add 0's to the end
        if current < length:
            wav = F.pad(wav, (0, length - current))
        # if curr length is too big, cut off extra samples
        elif current > length:
            wav = wav[:, :length]
        return wav

    def _decode_ref_pcm16_b64(self, b64_data: str, expected_sr: int) -> torch.Tensor:
        # take a base64 string of 16-bit audio, decode it, normalize it, and turn into
        # a pytorch waveform tensor
        if expected_sr != MODEL_SR:
            raise ValueError(f"ref_sr must be {MODEL_SR}, got {expected_sr}")

        raw = base64.b64decode(b64_data.encode("ascii"))
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            raise ValueError("reference pcm payload is empty")
        return torch.from_numpy(samples).unsqueeze(0)

    def compute_embedding(self, wav_cpu: torch.Tensor) -> torch.Tensor:
        # does the ecapa-tdnn embedding for curr string of audio
        # and computes the l2 normalization
        wav_dev = wav_cpu.to(self.device).float()
        with torch.no_grad():
            emb = self.spk_model.encode_batch(wav_dev).squeeze()
        if emb.ndim == 0:
            emb = emb.unsqueeze(0)
        norm = emb.norm(p=2)
        if float(norm.item()) < 1e-8:
            return torch.zeros_like(emb)
        emb = emb / norm
        return emb

    def _similarity(self, pred_full_cpu: torch.Tensor, ref_emb: torch.Tensor) -> float:
        if float(pred_full_cpu.abs().max().item()) < 1e-6:
            return 0.0
        if float(ref_emb.abs().max().item()) < 1e-8:
            return 0.0
        pred_dev = pred_full_cpu.to(self.device).float()
        with torch.no_grad():
            pred_emb = self.spk_model.encode_batch(pred_dev).squeeze()
        if pred_emb.ndim == 0:
            pred_emb = pred_emb.unsqueeze(0)
        norm = pred_emb.norm(p=2)
        if float(norm.item()) < 1e-8:
            return 0.0
        pred_emb = pred_emb / norm
        sim = F.cosine_similarity(pred_emb.unsqueeze(0), ref_emb.unsqueeze(0)).item()
        return float(sim)

    def extract_for_target(self, mix_context_cpu: torch.Tensor, ref_wav_cpu: torch.Tensor) -> torch.Tensor:
        pred = self.extractor.extract_speech_from_pcm(mix_context_cpu, MODEL_SR, ref_wav_cpu, MODEL_SR)
        if pred is None:
            pred = torch.zeros_like(mix_context_cpu)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        if pred.shape[0] > 1:
            pred = pred[:1]
        pred = self._fit_to_length(pred, mix_context_cpu.shape[1]).float().cpu()
        return pred

    def parse_targets(self, init_msg: dict) -> list[dict]:
        entries = init_msg.get("targets", [])
        if not isinstance(entries, list):
            raise ValueError("targets must be a list")

        parsed = []
        for i, entry in enumerate(entries[:MAX_TARGETS]):
            if not isinstance(entry, dict):
                continue

            try:
                target_id = str(entry.get("id", f"target{i}"))
                ref_sr = int(entry.get("ref_sr", MODEL_SR))
                ref_b64 = entry.get("ref_pcm16_b64", "")
                if not ref_b64:
                    continue

                ref_wav = self._decode_ref_pcm16_b64(ref_b64, ref_sr)
                ref_emb = self.compute_embedding(ref_wav)
                parsed.append(
                    {
                        "id": target_id,
                        "ref_wav": ref_wav,
                        "ref_emb": ref_emb,
                        "is_active": False,
                        "last_active_ts": 0.0,
                        "last_similarity": 0.0,
                    }
                )
            except Exception as target_parse_err:
                print(f"skip invalid target[{i}]: {target_parse_err}", flush=True)

        return parsed

    @modal.asgi_app()
    def serve(self):
        """Bind the FastAPI app and endpoints that use the class state natively."""
        web_app = FastAPI(title="Spatial Audio TSE (WeSep) - Native App")

        @web_app.get("/health")
        async def health():
            return {
                "status": "ok" if getattr(self, "extractor", None) is not None and getattr(self, "spk_model", None) is not None else "models_not_loaded",
                "cuda": torch.cuda.is_available() if hasattr(torch, "cuda") else False,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                "error": getattr(self, "load_error", None),
            }

        @web_app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            print("client connected", flush=True)

            if getattr(self, "extractor", None) is None or getattr(self, "spk_model", None) is None:
                await websocket.send_text(json.dumps({"type": "error", "detail": f"Models not loaded: {getattr(self, 'load_error', 'Unknown Error')}"}))
                await websocket.close()
                return

            context_buf = torch.zeros(1, CONTEXT_SAMPLES, dtype=torch.float32)
            chunks_received = 0

            try:
                init_raw = await websocket.receive_text()
                init_msg = json.loads(init_raw)
                if init_msg.get("type") != "init_tse_v1":
                    raise ValueError("expected init_tse_v1")

                targets = self.parse_targets(init_msg)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "ready",
                            "pipeline": "wesep_tse",
                            "targets_loaded": len(targets),
                            "sample_rate": MODEL_SR,
                            "chunk_samples": CHUNK_SAMPLES,
                        }
                    )
                )
                print(f"loaded targets={len(targets)}", flush=True)

                while True:
                    msg = await websocket.receive()

                    if "text" in msg and msg["text"]:
                        # Optional in-stream hot-reload support.
                        try:
                            reload_msg = json.loads(msg["text"])
                            if reload_msg.get("type") == "init_tse_v1":
                                targets = self.parse_targets(reload_msg)
                                await websocket.send_text(
                                    json.dumps(
                                        {
                                            "type": "ready",
                                            "pipeline": "wesep_tse",
                                            "targets_loaded": len(targets),
                                            "sample_rate": MODEL_SR,
                                            "chunk_samples": CHUNK_SAMPLES,
                                        }
                                    )
                                )
                                print(f"hot-reload targets={len(targets)}", flush=True)
                        except Exception as reload_err:
                            await websocket.send_text(json.dumps({"type": "error", "detail": f"reload failed: {reload_err}"}))
                        continue

                    raw = msg.get("bytes") if isinstance(msg, dict) else None
                    if not raw:
                        continue

                    t0 = time.perf_counter()

                    chunk_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    chunk_tensor = torch.from_numpy(chunk_np).unsqueeze(0)
                    chunk_len = int(chunk_tensor.shape[1])

                    if chunk_len >= CONTEXT_SAMPLES:
                        context_buf = chunk_tensor[:, -CONTEXT_SAMPLES:]
                    else:
                        context_buf = torch.cat([context_buf[:, chunk_len:], chunk_tensor], dim=1)

                    chunks_received += 1

                    if chunks_received < 4:
                        silence = np.zeros(chunk_len, dtype=np.int16)
                        await websocket.send_bytes(silence.tobytes())
                        continue

                    if not targets:
                        silence = np.zeros(chunk_len, dtype=np.int16)
                        await websocket.send_bytes(silence.tobytes())
                        if chunks_received % STATS_EVERY_N_CHUNKS == 0:
                            elapsed_ms = (time.perf_counter() - t0) * 1000.0
                            await websocket.send_text(
                                json.dumps(
                                    {
                                        "type": "stats",
                                        "active_ids": [],
                                        "top_similarity": 0.0,
                                        "process_ms": elapsed_ms,
                                    }
                                )
                            )
                        continue

                    # Sequential fallback per target.
                    mix_out = torch.zeros(chunk_len, dtype=torch.float32)
                    active_ids: list[str] = []
                    top_similarity = 0.0
                    now_ts = time.time()

                    for t in targets:
                        try:
                            pred_full = self.extract_for_target(context_buf, t["ref_wav"])
                            sim = self._similarity(pred_full, t["ref_emb"])
                        except Exception as target_err:
                            print(f"target={t['id']} error={target_err}", flush=True)
                            sim = 0.0
                            pred_full = torch.zeros_like(context_buf)

                        t["last_similarity"] = sim
                        top_similarity = max(top_similarity, sim)

                        if sim > SIMILARITY_THRESHOLD:
                            t["is_active"] = True
                            t["last_active_ts"] = now_ts
                        elif t["is_active"] and sim > HANG_THRESHOLD:
                            t["is_active"] = True
                            t["last_active_ts"] = now_ts
                        else:
                            if now_ts - t["last_active_ts"] < HANG_TIME:
                                t["is_active"] = True
                            else:
                                t["is_active"] = False

                        if t["is_active"]:
                            active_ids.append(t["id"])
                            mix_out += pred_full[0, -chunk_len:]

                    mix_out = torch.clamp(mix_out, -1.0, 1.0)
                    out_i16 = (mix_out * 32767.0).to(torch.int16).cpu().numpy()
                    await websocket.send_bytes(out_i16.tobytes())

                    if chunks_received % STATS_EVERY_N_CHUNKS == 0:
                        elapsed_ms = (time.perf_counter() - t0) * 1000.0
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "stats",
                                    "active_ids": active_ids,
                                    "top_similarity": top_similarity,
                                    "process_ms": elapsed_ms,
                                }
                            )
                        )

            except WebSocketDisconnect:
                print("client disconnected", flush=True)
            except Exception as exc:
                print(f"streaming error: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                try:
                    await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}))
                except Exception:
                    pass
            finally:
                print("connection cleanup complete", flush=True)

        return web_app

@app.local_entrypoint()
def main():
    print("Modal app 'spatial-audio-tse' (understanding_tse) is defined.")
    print("  deploy: modal deploy understanding_tse.py")
    print("  dev:    modal serve  understanding_tse.py")
