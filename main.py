"""
Unified Main Application (Fully Local WeSep TSE)
================================================

Combines:
  1. Real-time audio streaming to a local WebSocket server (WeSep TSE gating)
  2. Camera face tracking UI
  3. 'i' to record (5s) -> Dolphin + ECAPA-TDNN -> live WebSocket reconnect

Run:
  uv run python main.py --device auto
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn.functional as F
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy.signal import resample_poly


# =====================================================================
# Configuration
# =====================================================================

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_AVTSE_DIR = os.path.join(_REPO_ROOT, "av-tse")
_DOLPHIN_DIR = os.path.join(_REPO_ROOT, "Dolphin")
_WESEP_LOCAL_DIR = os.path.join(_REPO_ROOT, "third_party", "wesep")
_WESPEAKER_LOCAL_DIR = os.path.join(_REPO_ROOT, "third_party", "wespeaker")

MODEL_SR = 16_000
CHUNK_DURATION = 0.15
CHUNK_SAMPLES = int(MODEL_SR * CHUNK_DURATION)
CONTEXT_DURATION = 1.0
CONTEXT_SAMPLES = int(MODEL_SR * CONTEXT_DURATION)
REF_DURATION_SEC = 5.0
REF_SAMPLES = int(MODEL_SR * REF_DURATION_SEC)
MAX_TARGETS = 8

RECORD_FPS = 25
MAX_RECORD_SEC = 5

JITTER_BUFFER_SIZE = 2
EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "noise_gate", "embeddings")
REFERENCES_DIR = os.path.join(_REPO_ROOT, "noise_gate", "references")

SIMILARITY_THRESHOLD = 0.35
HANG_THRESHOLD = 0.25
HANG_TIME = 0.8
STATS_EVERY_N_CHUNKS = 10

# Global State
is_recording = False
record_start_time = 0.0
recorded_frames: list[np.ndarray] = []
recorded_audio_chunks: list[np.ndarray] = []
record_lock = threading.Lock()
reload_requested = threading.Event()
shutdown_requested = threading.Event()


@dataclass
class AppConfig:
    host: str
    port: int
    device: str
    cuda_device: int | None
    use_status: bool
    camera: int | None
    max_targets: int


APP_CONFIG: AppConfig | None = None


class ReconnectRequested(RuntimeError):
    """Raised to break out of websocket gather and reconnect."""


# =====================================================================
# Device helpers
# =====================================================================

def _is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _is_mps_backend_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    mps_markers = [
        "mps",
        "metal",
        "not implemented",
        "unsupported",
        "invalid type: 'torch.mps",
    ]
    return any(marker in message for marker in mps_markers)


def resolve_compute_device(requested_device: str, cuda_device: int | None = None) -> torch.device:
    requested = (requested_device or "auto").lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device '{requested_device}'. Use one of: auto, cpu, mps, cuda.")

    if requested in {"auto", "cuda"} and cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not _is_mps_available():
            raise RuntimeError("MPS was requested but is not available on this machine.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch runtime.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device_with_mps_fallback(requested_device: str, cuda_device: int | None = None) -> torch.device:
    try:
        return resolve_compute_device(requested_device, cuda_device)
    except Exception as exc:
        if requested_device == "mps":
            print(f"[Device] MPS unavailable ({exc}); falling back to CPU", flush=True)
            return torch.device("cpu")
        raise


# =====================================================================
# Target / Reference Utilities
# =====================================================================

def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        return audio.mean(axis=1, dtype=np.float32)
    return np.asarray(audio).reshape(-1).astype(np.float32)


def _resample_to_model_sr(audio: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == MODEL_SR:
        return audio.astype(np.float32, copy=False)
    g = math.gcd(sr_in, MODEL_SR)
    up = MODEL_SR // g
    down = sr_in // g
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def _fit_or_pad(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def _read_reference_16k(path: str, fixed_len: int | None = None) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = _to_mono_float32(wav)
    wav = _resample_to_model_sr(wav, sr)
    if fixed_len is not None:
        wav = _fit_or_pad(wav, fixed_len)
    return wav


def _encode_reference_payload(path: str) -> dict:
    wav = _read_reference_16k(path, fixed_len=REF_SAMPLES)
    pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16)
    return {
        "ref_sr": MODEL_SR,
        "ref_samples": int(pcm16.shape[0]),
        "ref_pcm16_b64": base64.b64encode(pcm16.tobytes()).decode("ascii"),
    }


def _save_reference_for_user(source_wav: str, user_id: str) -> str:
    os.makedirs(REFERENCES_DIR, exist_ok=True)
    ref_audio = _read_reference_16k(source_wav, fixed_len=None)
    out_path = os.path.join(REFERENCES_DIR, f"{user_id}.wav")
    sf.write(out_path, ref_audio, MODEL_SR)
    return out_path


def load_targets() -> list[dict]:
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(REFERENCES_DIR, exist_ok=True)

    npy_files = sorted(f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy"))
    targets: list[dict] = []
    missing_refs: list[str] = []

    for f in npy_files:
        stem = f[:-4]
        ref_path = os.path.join(REFERENCES_DIR, f"{stem}.wav")
        if not os.path.exists(ref_path):
            missing_refs.append(stem)
            continue

        vec = np.load(os.path.join(EMBEDDINGS_DIR, f))
        targets.append(
            {
                "id": stem,
                "embedding": vec.tolist(),
                "ref_wav_path": ref_path,
            }
        )
        if len(targets) >= MAX_TARGETS:
            break

    if missing_refs:
        print(
            "[Targets] Skipping targets with missing references: " + ", ".join(missing_refs),
            flush=True,
        )

    return targets


def build_init_payload() -> tuple[dict, list[str]]:
    targets = load_targets()
    transport_targets: list[dict] = []

    for t in targets:
        try:
            ref_payload = _encode_reference_payload(t["ref_wav_path"])
        except Exception as exc:
            print(f"[Targets] Failed to encode reference for {t['id']}: {exc}", flush=True)
            continue
        transport_targets.append({"id": t["id"], **ref_payload})

    msg = {
        "type": "init_tse_v1",
        "sample_rate": MODEL_SR,
        "chunk_samples": CHUNK_SAMPLES,
        "targets": transport_targets,
    }
    return msg, [t["id"] for t in targets]


def request_hot_reload() -> None:
    reload_requested.set()


# =====================================================================
# Local WeSep TSE server runtime
# =====================================================================

def _ensure_wesep_paths() -> None:
    for p in (_WESPEAKER_LOCAL_DIR, _WESEP_LOCAL_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


class LocalTSERuntime:
    def __init__(self, requested_device: str, cuda_device: int | None):
        self.requested_device = requested_device
        self.cuda_device = cuda_device
        self.device = torch.device("cpu")
        self.extractor = None
        self.spk_model = None
        self.load_error: str | None = None
        self._model_lock = threading.RLock()

    def initialize(self) -> None:
        self._load_with_fallback(self.requested_device)

    def _load_with_fallback(self, requested_device: str) -> None:
        try:
            resolved = resolve_compute_device(requested_device, self.cuda_device)
            self._load_models_on_device(str(resolved))
            return
        except Exception as exc:
            if requested_device == "mps" and _is_mps_backend_error(exc):
                print(f"[LocalTSE] MPS load failed ({exc}); retrying on CPU", flush=True)
                self._load_models_on_device("cpu")
                return
            self.load_error = str(exc)
            raise

    def _load_models_on_device(self, device_str: str) -> None:
        _ensure_wesep_paths()
        from speechbrain.inference.speaker import EncoderClassifier
        from wesep import load_model

        with self._model_lock:
            self.load_error = None
            self.device = torch.device(device_str)

            if self.device.type == "cuda" and self.cuda_device is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device)

            print(f"[LocalTSE] Loading WeSep+ECAPA on {self.device}", flush=True)

            extractor = load_model("english")
            extractor.set_device(device_str)
            extractor.set_resample_rate(MODEL_SR)
            extractor.set_vad(False)
            extractor.set_wavform_norm(True)
            extractor.set_output_norm(False)

            spk_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(_REPO_ROOT, "pretrained_models", "spkrec-ecapa-voxceleb"),
                run_opts={"device": device_str},
            )
            self.extractor = extractor
            self.spk_model = spk_model
            print("[LocalTSE] Models loaded", flush=True)

    def fallback_to_cpu_if_mps_error(self, exc: BaseException) -> bool:
        if self.device.type != "mps":
            return False
        if not _is_mps_backend_error(exc):
            return False
        try:
            print(f"[LocalTSE] Runtime MPS error ({exc}); switching to CPU", flush=True)
            self._load_models_on_device("cpu")
            return True
        except Exception as fallback_exc:
            self.load_error = str(fallback_exc)
            print(f"[LocalTSE] CPU fallback failed: {fallback_exc}", flush=True)
            return False

    def _fit_to_length(self, wav: torch.Tensor, length: int) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav[:1]
        current = wav.shape[1]
        if current < length:
            wav = F.pad(wav, (0, length - current))
        elif current > length:
            wav = wav[:, :length]
        return wav

    def decode_ref_pcm16_b64(self, b64_data: str, expected_sr: int) -> torch.Tensor:
        if expected_sr != MODEL_SR:
            raise ValueError(f"ref_sr must be {MODEL_SR}, got {expected_sr}")
        raw = base64.b64decode(b64_data.encode("ascii"))
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            raise ValueError("reference pcm payload is empty")
        return torch.from_numpy(samples).unsqueeze(0)

    def compute_embedding(self, wav_cpu: torch.Tensor) -> torch.Tensor:
        if self.spk_model is None:
            raise RuntimeError("speaker model is not loaded")
        wav_dev = wav_cpu.to(self.device).float()
        with self._model_lock, torch.no_grad():
            emb = self.spk_model.encode_batch(wav_dev).squeeze()
        if emb.ndim == 0:
            emb = emb.unsqueeze(0)
        norm = emb.norm(p=2)
        if float(norm.item()) < 1e-8:
            return torch.zeros_like(emb)
        emb = emb / norm
        return emb

    def similarity(self, pred_full_cpu: torch.Tensor, ref_emb: torch.Tensor) -> float:
        if self.spk_model is None:
            raise RuntimeError("speaker model is not loaded")
        if float(pred_full_cpu.abs().max().item()) < 1e-6:
            return 0.0
        if float(ref_emb.abs().max().item()) < 1e-8:
            return 0.0
        pred_dev = pred_full_cpu.to(self.device).float()
        with self._model_lock, torch.no_grad():
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
        if self.extractor is None:
            raise RuntimeError("extractor is not loaded")
        with self._model_lock:
            pred = self.extractor.extract_speech_from_pcm(mix_context_cpu, MODEL_SR, ref_wav_cpu, MODEL_SR)
        if pred is None:
            pred = torch.zeros_like(mix_context_cpu)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        if pred.shape[0] > 1:
            pred = pred[:1]
        pred = self._fit_to_length(pred, mix_context_cpu.shape[1]).float().cpu()
        return pred

    def parse_targets(self, init_msg: dict[str, Any]) -> list[dict[str, Any]]:
        entries = init_msg.get("targets", [])
        if not isinstance(entries, list):
            raise ValueError("targets must be a list")

        parsed: list[dict[str, Any]] = []
        for i, entry in enumerate(entries[:MAX_TARGETS]):
            if not isinstance(entry, dict):
                continue
            try:
                target_id = str(entry.get("id", f"target{i}"))
                ref_sr = int(entry.get("ref_sr", MODEL_SR))
                ref_b64 = entry.get("ref_pcm16_b64", "")
                if not ref_b64:
                    continue
                ref_wav = self.decode_ref_pcm16_b64(ref_b64, ref_sr)
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
                print(f"[LocalTSE] skip invalid target[{i}]: {target_parse_err}", flush=True)
        return parsed


SERVER_RUNTIME: LocalTSERuntime | None = None
SERVER_CONTROLLER: "UvicornServerController | None" = None

web_app = FastAPI(title="Local Spatial Audio TSE (WeSep)")


@web_app.get("/health")
async def health() -> dict[str, Any]:
    if SERVER_RUNTIME is None:
        return {"status": "not_initialized"}
    return {
        "status": "ok" if SERVER_RUNTIME.extractor is not None and SERVER_RUNTIME.spk_model is not None else "models_not_loaded",
        "device": str(SERVER_RUNTIME.device),
        "error": SERVER_RUNTIME.load_error,
    }


@web_app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[LocalTSE] client connected", flush=True)

    if SERVER_RUNTIME is None:
        await websocket.send_text(json.dumps({"type": "error", "detail": "Server runtime not initialized"}))
        await websocket.close()
        return
    runtime = SERVER_RUNTIME

    if runtime.extractor is None or runtime.spk_model is None:
        await websocket.send_text(
            json.dumps({"type": "error", "detail": f"Models not loaded: {runtime.load_error}"})
        )
        await websocket.close()
        return

    context_buf = torch.zeros(1, CONTEXT_SAMPLES, dtype=torch.float32)
    chunks_received = 0

    try:
        init_raw = await websocket.receive_text()
        init_msg = json.loads(init_raw)
        if init_msg.get("type") != "init_tse_v1":
            raise ValueError("expected init_tse_v1")

        targets = runtime.parse_targets(init_msg)
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
        print(f"[LocalTSE] loaded targets={len(targets)}", flush=True)

        while True:
            msg = await websocket.receive()

            if "text" in msg and msg["text"]:
                try:
                    reload_msg = json.loads(msg["text"])
                    if reload_msg.get("type") == "init_tse_v1":
                        targets = runtime.parse_targets(reload_msg)
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
                        print(f"[LocalTSE] hot-reload targets={len(targets)}", flush=True)
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

            mix_out = torch.zeros(chunk_len, dtype=torch.float32)
            active_ids: list[str] = []
            top_similarity = 0.0
            now_ts = time.time()

            for t in targets:
                try:
                    pred_full = runtime.extract_for_target(context_buf, t["ref_wav"])
                    sim = runtime.similarity(pred_full, t["ref_emb"])
                except Exception as target_err:
                    if runtime.fallback_to_cpu_if_mps_error(target_err):
                        pred_full = runtime.extract_for_target(context_buf, t["ref_wav"])
                        sim = runtime.similarity(pred_full, t["ref_emb"])
                    else:
                        print(f"[LocalTSE] target={t['id']} error={target_err}", flush=True)
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
        print("[LocalTSE] client disconnected", flush=True)
    except Exception as exc:
        print(f"[LocalTSE] streaming error: {type(exc).__name__}: {exc}", flush=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        except Exception:
            pass
    finally:
        print("[LocalTSE] connection cleanup complete", flush=True)


class UvicornServerController:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None
        self._server: uvicorn.Server | None = None
        self._started = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._started.wait(timeout=10.0)

    def _run(self) -> None:
        config = uvicorn.Config(web_app, host=self.host, port=self.port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        self._started.set()
        self._server.run()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)


# =====================================================================
# Background Processing (Dolphin -> ECAPA)
# =====================================================================

def _status_callback(message: dict[str, Any]) -> None:
    status = message.get("status", "")
    progress = message.get("progress")
    if progress is None:
        print(f"[Dolphin] {status}", flush=True)
    else:
        print(f"[Dolphin {float(progress) * 100:5.1f}%] {status}", flush=True)


def _run_dolphin_inference(
    video_path: str,
    output_dir: str,
    speakers: int,
    cuda_device: int | None,
    requested_device: str,
    use_status: bool,
) -> list[str]:
    if _DOLPHIN_DIR not in sys.path:
        sys.path.insert(0, _DOLPHIN_DIR)

    from Inference import process_video, resolve_device

    try:
        from Inference_with_status import process_video_with_status
    except Exception:
        process_video_with_status = None

    selected_device = resolve_device(requested_device, cuda_device)
    print(f"[Dolphin] Requested={requested_device} Selected={selected_device}", flush=True)

    if use_status and process_video_with_status is not None:
        print("[Dolphin] Using process_video_with_status", flush=True)
        return process_video_with_status(
            input_file=video_path,
            output_path=output_dir,
            number_of_speakers=speakers,
            detect_every_N_frame=8,
            scalar_face_detection=1.5,
            cuda_device=cuda_device,
            device=requested_device,
            status_callback=_status_callback,
        )

    if use_status and process_video_with_status is None:
        print("[Dolphin] Inference_with_status unavailable; using process_video", flush=True)
    else:
        print("[Dolphin] Using process_video", flush=True)
    return process_video(
        input_file=video_path,
        output_path=output_dir,
        number_of_speakers=speakers,
        detect_every_N_frame=8,
        scalar_face_detection=1.5,
        cuda_device=cuda_device,
        device=requested_device,
    )


def _process_recording_thread(video_frames: list[np.ndarray], audio_chunks: list[np.ndarray], duration: float):
    global APP_CONFIG
    if APP_CONFIG is None:
        print("[Processing] Missing app config", flush=True)
        return

    print("\n[Processing] Starting background extraction and enrollment...")

    temp_dir = tempfile.mkdtemp()
    output_mp4 = os.path.join(temp_dir, "recording.mp4")

    n_frames = len(video_frames)
    h, w = video_frames[0].shape[:2]
    actual_fps = n_frames / duration if duration > 0 else RECORD_FPS

    print(f"[Processing] Saving MP4 ({w}x{h}) @ {actual_fps:.1f} fps")
    tmp_video = os.path.join(temp_dir, "v.mp4")
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, actual_fps, (w, h))
    for frame in video_frames:
        writer.write(frame)
    writer.release()

    raw_audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.zeros((1, 1), dtype=np.float32)
    if len(raw_audio.shape) > 1 and raw_audio.shape[1] > 1:
        mono_16k = raw_audio.mean(axis=1)
    else:
        mono_16k = raw_audio.reshape(-1)

    target_samples = int(duration * MODEL_SR)
    if len(mono_16k) < target_samples:
        mono_16k = np.pad(mono_16k, (0, target_samples - len(mono_16k)))
    else:
        mono_16k = mono_16k[:target_samples]
    mono_16k = mono_16k.astype(np.float32)

    tmp_audio = os.path.join(temp_dir, "a.wav")
    sf.write(tmp_audio, mono_16k, MODEL_SR)

    os.system(
        f'ffmpeg -y -i "{tmp_video}" -i "{tmp_audio}" '
        f'-c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 128k '
        f'-shortest "{output_mp4}" -loglevel warning'
    )

    dolphin_out = os.path.join(temp_dir, "separated")
    wav_to_enroll = None

    try:
        print("[Processing] Running Dolphin locally for separation...")
        requested_device = APP_CONFIG.device
        cuda_device = APP_CONFIG.cuda_device
        try:
            _run_dolphin_inference(
                video_path=output_mp4,
                output_dir=dolphin_out,
                speakers=1,
                cuda_device=cuda_device,
                requested_device=requested_device,
                use_status=APP_CONFIG.use_status,
            )
        except Exception as exc:
            if requested_device == "mps" and _is_mps_backend_error(exc):
                print(f"[Processing] Dolphin MPS error ({exc}); retrying on CPU", flush=True)
                _run_dolphin_inference(
                    video_path=output_mp4,
                    output_dir=dolphin_out,
                    speakers=1,
                    cuda_device=None,
                    requested_device="cpu",
                    use_status=APP_CONFIG.use_status,
                )
            else:
                raise

        wav_to_enroll = os.path.join(dolphin_out, "speaker1_est.wav")
        if not os.path.exists(wav_to_enroll):
            raise RuntimeError("Dolphin ran but no speaker1_est.wav was emitted")
    except Exception as exc:
        print(f"[Processing] Dolphin unavailable/failing ({exc}); using raw audio")
        wav_to_enroll = tmp_audio

    print("[Processing] Running ECAPA-TDNN enrollment...")
    try:
        if _AVTSE_DIR not in sys.path:
            sys.path.insert(0, _AVTSE_DIR)
        from ecapa_enroll import enroll_from_wav

        enroll_device = str(_resolve_device_with_mps_fallback(APP_CONFIG.device, APP_CONFIG.cuda_device))
        try:
            emb_path = enroll_from_wav(wav_to_enroll, device=enroll_device)
        except Exception as exc:
            if enroll_device == "mps" and _is_mps_backend_error(exc):
                print(f"[Processing] ECAPA MPS error ({exc}); retrying on CPU", flush=True)
                emb_path = enroll_from_wav(wav_to_enroll, device="cpu")
            else:
                raise

        user_id = os.path.splitext(os.path.basename(emb_path))[0]
        ref_path = _save_reference_for_user(wav_to_enroll, user_id)

        print(f"[Processing] Enrollment complete: {emb_path}")
        print(f"[Processing] Saved reference wav: {ref_path}")

        request_hot_reload()
        print("[Processing] Requested websocket reconnect for target reload")
    except Exception as exc:
        print(f"[Processing] Enrollment failed: {exc}")


# =====================================================================
# OpenCV UI Thread (runs synchronously in main loop)
# =====================================================================

def run_camera_ui(camera_index: int | None = None):
    global is_recording, record_start_time, recorded_frames, recorded_audio_chunks

    if _AVTSE_DIR not in sys.path:
        sys.path.insert(0, _AVTSE_DIR)
    from targeting import LipTargetingSystem

    print(f"\n[UI] Opening targeting camera (index={camera_index if camera_index is not None else 'auto'})...")
    with LipTargetingSystem(camera_index=camera_index) as ts:
        while True:
            ok, frame = ts._cap.read()
            if not ok:
                break

            ts._process_frame(frame, draw=True)

            if is_recording:
                elapsed = time.time() - record_start_time
                cv2.putText(
                    frame,
                    f"REC {elapsed:.1f}s / {MAX_RECORD_SEC}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)

                if getattr(ts, "current_raw_frame", None) is not None:
                    recorded_frames.append(ts.current_raw_frame.copy())
                else:
                    recorded_frames.append(frame.copy())

                if elapsed >= MAX_RECORD_SEC:
                    print(f"\n[Recorder] Max duration reached ({MAX_RECORD_SEC}s). Stopping.")
                    is_recording = False
                    ts.is_locked = False

                    v_buf = list(recorded_frames)
                    with record_lock:
                        a_buf = list(recorded_audio_chunks)
                    threading.Thread(
                        target=_process_recording_thread,
                        args=(v_buf, a_buf, elapsed),
                        daemon=True,
                    ).start()
            else:
                cv2.putText(
                    frame,
                    "STANDBY - press 'i' to record",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Main - Camera + Local TSE Gating", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("i") and not is_recording:
                is_recording = True
                ts.is_locked = True
                record_start_time = time.time()
                recorded_frames.clear()
                with record_lock:
                    recorded_audio_chunks.clear()
                print("\n[Recorder] Recording started (5 seconds)...")
            elif key == ord("q") or key == 27:
                break

    cv2.destroyAllWindows()


# =====================================================================
# Asyncio Streaming Client
# =====================================================================

async def stream_audio_task(ws_url: str):
    global is_recording, recorded_audio_chunks

    if _AVTSE_DIR not in sys.path:
        sys.path.insert(0, _AVTSE_DIR)
    from record import _get_best_mic

    idx, chs = _get_best_mic()
    print(f"[Audio] Starting mic stream (input dev={idx}, output dev=default)")

    pending_audio: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()
    recv_queue = collections.deque(maxlen=32)

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"[Audio] {status}", flush=True)

        mono = indata[:, 0].copy()
        pending_audio.put(mono)

        if is_recording:
            with record_lock:
                recorded_audio_chunks.append(indata.copy())

        if len(recv_queue) > 0:
            while len(recv_queue) > JITTER_BUFFER_SIZE + 1:
                recv_queue.popleft()
            processed = recv_queue.popleft()
            if len(processed) >= frames:
                outdata[:, 0] = processed[:frames]
            else:
                outdata[:frames, 0] = 0.0
                outdata[: len(processed), 0] = processed
        else:
            outdata[:, 0] = 0.0
            stats["drops"] += 1

    while not shutdown_requested.is_set():
        init_payload, target_ids = build_init_payload()

        print(f"\n[WebSocket] Connecting to {ws_url}")
        print(
            f"[WebSocket] Targets loaded for init: {len(target_ids)} -> "
            f"{', '.join(target_ids) if target_ids else '(none)'}"
        )

        playback_started = False
        chunks_buffered = 0
        stats = {
            "sent": 0,
            "drops": 0,
            "active_ids": [],
            "top_similarity": 0.0,
            "process_ms": 0.0,
        }
        accum_buf = np.zeros(0, dtype=np.float32)
        recv_queue.clear()
        reload_requested.clear()

        while True:
            try:
                pending_audio.get_nowait()
            except queue.Empty:
                break

        async def producer(ws):
            nonlocal accum_buf
            while not shutdown_requested.is_set():
                if reload_requested.is_set():
                    reload_requested.clear()
                    print("\n[WebSocket] Target reload requested; reconnecting...")
                    await ws.close(code=4001, reason="hot_reload")
                    raise ReconnectRequested()

                drained = []
                while True:
                    try:
                        drained.append(pending_audio.get_nowait())
                    except queue.Empty:
                        break

                if drained:
                    accum_buf = np.concatenate([accum_buf, np.concatenate(drained)])

                while len(accum_buf) >= CHUNK_SAMPLES:
                    chunk = accum_buf[:CHUNK_SAMPLES]
                    accum_buf = accum_buf[CHUNK_SAMPLES:]
                    int16_data = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                    await ws.send(int16_data.tobytes())
                    stats["sent"] += 1

                if not drained:
                    await asyncio.sleep(0.005)

        async def consumer(ws):
            nonlocal playback_started, chunks_buffered
            while not shutdown_requested.is_set():
                raw = await ws.recv()

                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        print(f"\n[WebSocket] Ignoring text frame: {raw[:80]}")
                        continue

                    mtype = msg.get("type")
                    if mtype == "error":
                        print(f"\n[WebSocket] Server error: {msg.get('detail')}")
                    elif mtype == "stats":
                        stats["active_ids"] = msg.get("active_ids", [])
                        stats["top_similarity"] = float(msg.get("top_similarity", 0.0))
                        stats["process_ms"] = float(msg.get("process_ms", 0.0))
                    continue

                if not raw:
                    continue

                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                recv_queue.append(samples)

                if not playback_started:
                    chunks_buffered += 1
                    if chunks_buffered >= JITTER_BUFFER_SIZE:
                        playback_started = True

        async def status_printer():
            while not shutdown_requested.is_set():
                await asyncio.sleep(0.3)
                active_ids = stats["active_ids"]
                if active_ids:
                    indicator = f"ACTIVE[{len(active_ids)}]: {','.join(active_ids)}"
                else:
                    indicator = "silence"
                sys.stdout.write(
                    "\r[TSE] "
                    f"{indicator:<36} "
                    f"sim={stats['top_similarity']:.2f} "
                    f"proc={stats['process_ms']:.1f}ms "
                    f"sent={stats['sent']:4d} drops={stats['drops']:2d}   "
                )
                sys.stdout.flush()

        try:
            async with websockets.connect(
                ws_url,
                max_size=16 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=60,
                open_timeout=30,
            ) as ws:
                await ws.send(json.dumps(init_payload))
                ready_raw = await ws.recv()

                if not isinstance(ready_raw, str):
                    raise RuntimeError("Expected text ready frame from server")

                resp = json.loads(ready_raw)
                if resp.get("type") == "error":
                    print(f"\n[WebSocket] Server error: {resp.get('detail')}")
                    await asyncio.sleep(1.0)
                    continue

                if resp.get("type") != "ready":
                    raise RuntimeError(f"Unexpected ready message: {resp}")

                print(
                    f"[WebSocket] Ready pipeline={resp.get('pipeline')} "
                    f"targets_loaded={resp.get('targets_loaded')}"
                )

                stream_ctx = sd.Stream(
                    device=(idx, sd.default.device[1]),
                    samplerate=MODEL_SR,
                    blocksize=CHUNK_SAMPLES,
                    channels=(chs, 1),
                    dtype="float32",
                    callback=audio_callback,
                )

                with stream_ctx:
                    await asyncio.gather(producer(ws), consumer(ws), status_printer())

        except ReconnectRequested:
            continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if shutdown_requested.is_set():
                break
            print(f"\n[WebSocket] Disconnected ({exc}); reconnecting in 1s")
            await asyncio.sleep(1.0)


# =====================================================================
# Main Thread Entry
# =====================================================================

def _run_asyncio_thread(url: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(stream_audio_task(url))
    except Exception as exc:
        print(f"\n[WebSocket] Thread exited: {exc}")
    finally:
        loop.close()


def main() -> None:
    global APP_CONFIG, MAX_TARGETS, SERVER_RUNTIME, SERVER_CONTROLLER

    parser = argparse.ArgumentParser(description="Unified Local App (WeSep TSE)")
    parser.add_argument("--camera", type=int, default=None, help="Force a specific camera index (e.g., 0)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Compute device")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Local TSE server bind host")
    parser.add_argument("--port", type=int, default=8000, help="Local TSE server bind port")
    parser.add_argument("--no-status", dest="use_status", action="store_false", help="Disable Dolphin status callback")
    parser.add_argument("--max-targets", type=int, default=8, help="Max enrolled targets to load")
    parser.set_defaults(use_status=True)
    args = parser.parse_args()

    APP_CONFIG = AppConfig(
        host=args.host,
        port=args.port,
        device=args.device,
        cuda_device=args.cuda_device,
        use_status=args.use_status,
        camera=args.camera,
        max_targets=max(1, args.max_targets),
    )
    MAX_TARGETS = APP_CONFIG.max_targets
    os.makedirs(REFERENCES_DIR, exist_ok=True)

    try:
        SERVER_RUNTIME = LocalTSERuntime(requested_device=APP_CONFIG.device, cuda_device=APP_CONFIG.cuda_device)
        SERVER_RUNTIME.initialize()
    except Exception as exc:
        print(f"[Error] Failed to initialize local TSE runtime: {exc}")
        sys.exit(1)

    SERVER_CONTROLLER = UvicornServerController(host=APP_CONFIG.host, port=APP_CONFIG.port)
    SERVER_CONTROLLER.start()
    ws_url = f"ws://{APP_CONFIG.host}:{APP_CONFIG.port}/ws"
    print(f"[Main] Local TSE server listening at {ws_url}")
    print(f"[Main] Device selected: {SERVER_RUNTIME.device}")

    t = threading.Thread(target=_run_asyncio_thread, args=(ws_url,), daemon=True)
    t.start()

    try:
        time.sleep(1.0)
        run_camera_ui(camera_index=APP_CONFIG.camera)
    finally:
        shutdown_requested.set()
        t.join(timeout=2.0)
        if SERVER_CONTROLLER is not None:
            SERVER_CONTROLLER.stop()
        print("\n[Main] Exiting main")


if __name__ == "__main__":
    main()
