"""
Unified Main Application v2 (WeSep TSE)
=======================================

Combines:
  1. Real-time audio streaming to Modal (WeSep TSE gating)
  2. Camera face tracking UI
  3. 'i' to record (5s) -> Dolphin + ECAPA-TDNN -> live WebSocket reconnect

Run:
  uv run python main_modal.py wss://YOUR-MODAL-URL/ws
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import concurrent.futures
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets
from scipy.signal import resample_poly



# =====================================================================
# Configuration
# =====================================================================

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_SR = 16_000
CHUNK_DURATION = 0.15
CHUNK_SAMPLES = int(MODEL_SR * CHUNK_DURATION)
CONTEXT_DURATION = 1.0
CONTEXT_SAMPLES = int(MODEL_SR * CONTEXT_DURATION)
REF_DURATION_SEC = 5.0
REF_SAMPLES = int(MODEL_SR * REF_DURATION_SEC)
MAX_TARGETS = 8

# For the 5-second recording
RECORD_FPS = 25
MAX_RECORD_SEC = 5

JITTER_BUFFER_SIZE = 2
EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "noise_gate", "embeddings")
REFERENCES_DIR = os.path.join(_REPO_ROOT, "noise_gate", "references")

# Modal Dolphin options (overridden by CLI in main()).
DOLPHIN_APP = "dolphin-av-tse"
DOLPHIN_METHOD = "DolphinSeparator.separate"
DOLPHIN_TIMEOUT_SEC = 180
DOLPHIN_RETRIES = 2
DOLPHIN_SPEAKERS = 1

# Global State
is_recording = False
record_start_time = 0.0
recorded_frames: list[np.ndarray] = []
recorded_audio_chunks: list[np.ndarray] = []
record_lock = threading.Lock()
reload_requested = threading.Event()
shutdown_requested = threading.Event()


class ReconnectRequested(RuntimeError):
    """Raised to break out of websocket gather and reconnect."""


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
            "[Targets] Skipping targets with missing references: "
            + ", ".join(missing_refs),
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
    """Signal the streaming loop to reconnect and refresh enrolled targets."""
    reload_requested.set()


# =====================================================================
# Background Processing (Dolphin -> ECAPA)
# =====================================================================

def _validate_dolphin_results(results: object) -> bytes:
    if not isinstance(results, dict):
        raise TypeError(f"expected dict from Modal Dolphin, got {type(results).__name__}")
    if "speaker1" not in results:
        raise RuntimeError("Modal Dolphin response missing required key 'speaker1'")
    speaker1 = results["speaker1"]
    if not isinstance(speaker1, (bytes, bytearray)):
        raise TypeError(f"expected bytes for 'speaker1', got {type(speaker1).__name__}")
    if len(speaker1) == 0:
        raise RuntimeError("Modal Dolphin returned empty 'speaker1' bytes")
    return bytes(speaker1)


def _run_modal_dolphin(video_bytes: bytes) -> bytes:
    import modal

    total_attempts = 1 + max(0, int(DOLPHIN_RETRIES))
    print(
        f"[Processing] Modal Dolphin target={DOLPHIN_APP}.{DOLPHIN_METHOD} "
        f"speakers={DOLPHIN_SPEAKERS} attempts={total_attempts}",
        flush=True,
    )
    dolphin_fn = modal.Function.from_name(DOLPHIN_APP, DOLPHIN_METHOD)

    timeout_budget = float(max(1, DOLPHIN_TIMEOUT_SEC))
    deadline = time.time() + timeout_budget
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Modal Dolphin timed out after {timeout_budget:.1f}s total budget"
            ) from last_error

        print(
            f"[Processing] Modal Dolphin attempt {attempt}/{total_attempts} "
            f"(remaining budget={remaining:.1f}s)",
            flush=True,
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(dolphin_fn.remote, video_bytes, num_speakers=DOLPHIN_SPEAKERS)
                results = fut.result(timeout=max(1.0, remaining))
            speaker1_bytes = _validate_dolphin_results(results)
            print("[Processing] Modal Dolphin validation passed", flush=True)
            return speaker1_bytes
        except Exception as exc:
            last_error = exc
            print(f"[Processing] Modal Dolphin attempt {attempt} failed: {exc}", flush=True)
            if attempt < total_attempts:
                time.sleep(min(1.0 * attempt, 2.0))

    raise RuntimeError("Modal Dolphin failed after all attempts") from last_error

def _process_recording_thread(video_frames: list[np.ndarray], audio_chunks: list[np.ndarray], duration: float):
    print("\n[Processing] Starting background extraction and enrollment...")

    # 1. Save MP4
    temp_dir = tempfile.mkdtemp()
    output_mp4 = os.path.join(temp_dir, "recording.mp4")

    n_frames = len(video_frames)
    h, w = video_frames[0].shape[:2]
    actual_fps = n_frames / duration if duration > 0 else 30.0

    print(f"[Processing] Saving MP4 ({w}x{h}) @ {actual_fps:.1f} fps")
    tmp_video = os.path.join(temp_dir, "v.mp4")
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, actual_fps, (w, h))
    for frame in video_frames:
        writer.write(frame)
    writer.release()

    raw_audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.zeros((1, 1), dtype=np.float32)

    # Ensure mono
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

    # 2. Dolphin
    dolphin_out = os.path.join(temp_dir, "separated")
    wav_to_enroll = None

    try:
        with open(output_mp4, "rb") as f:
            video_bytes = f.read()

        speaker1_bytes = _run_modal_dolphin(video_bytes)
        wav_to_enroll = os.path.join(dolphin_out, "speaker1_est.wav")
        os.makedirs(dolphin_out, exist_ok=True)
        with open(wav_to_enroll, "wb") as f:
            f.write(speaker1_bytes)

        if not os.path.exists(wav_to_enroll) or os.path.getsize(wav_to_enroll) == 0:
            raise RuntimeError("speaker1_est.wav was not written correctly")

    except Exception as exc:
        print(f"[Processing] Dolphin unavailable/failing ({exc}); using raw audio")
        wav_to_enroll = tmp_audio

    # 3. ECAPA enrollment + reference sync
    print("[Processing] Running ECAPA-TDNN enrollment...")
    try:
        _AVTSE_DIR = os.path.join(_REPO_ROOT, "av-tse")
        if _AVTSE_DIR not in sys.path:
            sys.path.insert(0, _AVTSE_DIR)
        from ecapa_enroll import enroll_from_wav

        emb_path = enroll_from_wav(wav_to_enroll)
        user_id = os.path.splitext(os.path.basename(emb_path))[0]
        ref_path = _save_reference_for_user(wav_to_enroll, user_id)

        print(f"[Processing] Enrollment complete: {emb_path}")
        print(f"[Processing] Saved reference wav: {ref_path}")

        # 4. Signal stream reconnect so init payload includes new target
        request_hot_reload()
        print("[Processing] Requested websocket reconnect for target reload")
    except Exception as exc:
        print(f"[Processing] Enrollment failed: {exc}")


# =====================================================================
# OpenCV UI Thread (runs synchronously in main loop)
# =====================================================================

def run_camera_ui(camera_index: int | None = None):
    global is_recording, record_start_time, recorded_frames, recorded_audio_chunks

    # Import inside function to prevent multiprocessing spawn thread from
    # initializing mediapipe/tensorflow and breaking RetinaFace logic.
    _REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    _AVTSE_DIR = os.path.join(_REPO_ROOT, "av-tse")
    if _AVTSE_DIR not in sys.path:
        sys.path.insert(0, _AVTSE_DIR)
    from targeting import LipTargetingSystem

    print(f"\n[UI] Opening targeting camera (index={camera_index if camera_index is not None else 'auto'})...")
    with LipTargetingSystem(camera_index=camera_index) as ts:
        while True:
            ok, frame = ts._cap.read()
            if not ok:
                break

            # Handles MediaPipe tracking
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

            cv2.imshow("Main App2 - Camera + TSE Gating", frame)

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

    # Import inside function to prevent multiprocessing spawn thread issues
    _REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    _AVTSE_DIR = os.path.join(_REPO_ROOT, "av-tse")
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

        # 1. Provide input to websocket buffer
        mono = indata[:, 0].copy()
        pending_audio.put(mono)

        # 2. Record raw input while enrollment capture is active
        if is_recording:
            with record_lock:
                recorded_audio_chunks.append(indata.copy())

        # 3. Playback from websocket receive buffer
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
        print(f"[WebSocket] Targets loaded for init: {len(target_ids)} -> {', '.join(target_ids) if target_ids else '(none)'}")

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
            # Triggered by hot reload event; immediately reconnect.
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
    global DOLPHIN_APP, DOLPHIN_METHOD, DOLPHIN_TIMEOUT_SEC, DOLPHIN_RETRIES, DOLPHIN_SPEAKERS

    parser = argparse.ArgumentParser(description="Unified App v2 (WeSep TSE)")
    parser.add_argument("url", nargs="?", default=os.environ.get("MODAL_WS_URL", ""))
    parser.add_argument("--camera", type=int, default=None, help="Force a specific camera index (e.g., 0)")
    parser.add_argument("--dolphin-app", default="dolphin-av-tse", help="Modal app name for Dolphin")
    parser.add_argument(
        "--dolphin-method",
        default="DolphinSeparator.separate",
        help="Modal method name for Dolphin",
    )
    parser.add_argument(
        "--dolphin-timeout-sec",
        type=int,
        default=180,
        help="Total timeout budget for Modal Dolphin call attempts",
    )
    parser.add_argument(
        "--dolphin-retries",
        type=int,
        default=2,
        help="Retries after the first Modal Dolphin call",
    )
    parser.add_argument(
        "--dolphin-speakers",
        type=int,
        default=1,
        help="num_speakers passed to Modal Dolphin",
    )
    args = parser.parse_args()

    if not args.url:
        print("[Error] No WebSocket URL provided")
        sys.exit(1)

    url = args.url.rstrip("/")
    if not url.endswith("/ws"):
        url += "/ws"

    DOLPHIN_APP = args.dolphin_app
    DOLPHIN_METHOD = args.dolphin_method
    DOLPHIN_TIMEOUT_SEC = max(1, int(args.dolphin_timeout_sec))
    DOLPHIN_RETRIES = max(0, int(args.dolphin_retries))
    DOLPHIN_SPEAKERS = max(1, int(args.dolphin_speakers))
    print(
        f"[Main] Dolphin Modal target={DOLPHIN_APP}.{DOLPHIN_METHOD} "
        f"speakers={DOLPHIN_SPEAKERS} timeout={DOLPHIN_TIMEOUT_SEC}s retries={DOLPHIN_RETRIES}",
        flush=True,
    )

    os.makedirs(REFERENCES_DIR, exist_ok=True)

    t = threading.Thread(target=_run_asyncio_thread, args=(url,), daemon=True)
    t.start()

    time.sleep(1.0)
    run_camera_ui(camera_index=args.camera)
    shutdown_requested.set()

    t.join(timeout=2.0)
    print("\n[Main] Exiting main_modal")


if __name__ == "__main__":
    main()
