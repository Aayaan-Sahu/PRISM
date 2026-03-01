"""
Unified Main Application
========================

Combines:
  1. Real-time audio streaming to Modal (noise gating)
  2. Camera face tracking UI
  3. 'i' to record (5s) -> Dolphin + ECAPA-TDNN -> live update WebSocket

Run:
  uv run python main_app.py wss://YOUR-MODAL-URL/ws
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import queue
import struct
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

# Ensure av-tse is importable
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_AVTSE_DIR = os.path.join(_REPO_ROOT, "av-tse")
if _AVTSE_DIR not in sys.path:
    sys.path.insert(0, _AVTSE_DIR)

from record import _get_best_mic
from targeting import LipTargetingSystem

# =====================================================================
# Configuration
# =====================================================================

MODEL_SR = 16_000
CHUNK_DURATION = 0.15
CHUNK_SAMPLES = int(MODEL_SR * CHUNK_DURATION)

# For the 5-second recording
RECORD_FPS = 25
MAX_RECORD_SEC = 5

JITTER_BUFFER_SIZE = 2
EMBEDDINGS_DIR = os.path.join(_REPO_ROOT, "noise_gate", "embeddings")

# Global State
is_recording = False
record_start_time = 0.0
recorded_frames = []
recorded_audio_chunks = []
record_lock = threading.Lock()
reload_requested = threading.Event()
shutdown_requested = threading.Event()

# =====================================================================
# Load Embeddings
# =====================================================================

def load_embeddings() -> tuple[list[list[float]], list[str]]:
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        return [], []

    npy_files = sorted(f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy"))
    if not npy_files:
        return [], []

    embeddings = []
    names = []
    for f in npy_files:
        vec = np.load(os.path.join(EMBEDDINGS_DIR, f))
        embeddings.append(vec.tolist())
        names.append(f[:-4])

    return embeddings, names


def request_hot_reload():
    """Signal the streaming loop to refresh enrolled embeddings."""
    reload_requested.set()

# =====================================================================
# Background Processing (Dolphin -> ECAPA)
# =====================================================================

def _process_recording_thread(video_frames: list, audio_chunks: list, duration: float):
    print("\n[Processing] Starting background extraction & enrollment...")
    
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
    for f in video_frames: writer.write(f)
    writer.release()

    raw_audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.zeros((1, 1))
    
    # Ensure Mono
    if len(raw_audio.shape) > 1 and raw_audio.shape[1] > 1:
        mono_16k = raw_audio.mean(axis=1)
    else:
        mono_16k = raw_audio.reshape(-1)
        
    # The stream is already 16kHz, so NO decimation needed.
    mono_16k = mono_16k[:int(duration * MODEL_SR)].astype(np.float32)
    
    tmp_audio = os.path.join(temp_dir, "a.wav")
    sf.write(tmp_audio, mono_16k, MODEL_SR)

    os.system(f'ffmpeg -y -i "{tmp_video}" -i "{tmp_audio}" -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 128k -shortest "{output_mp4}" -loglevel warning')
    
    # 2. Dolphin
    dolphin_out = os.path.join(temp_dir, "separated")
    wav_to_enroll = None
    
    try:
        # Check if Dolphin exists
        sys.path.insert(0, os.path.join(_REPO_ROOT, "Dolphin"))
        from Inference import process_video
        print("[Processing] Running Dolphin speech separation...")
        process_video(
            input_file=output_mp4,
            output_path=dolphin_out,
            number_of_speakers=1,
            detect_every_N_frame=8,
            scalar_face_detection=1.5,
            cuda_device=None,
        )
        wav_to_enroll = os.path.join(dolphin_out, "speaker1_est.wav")
        if not os.path.exists(wav_to_enroll):
            raise FileNotFoundError("Dolphin ran but no speaker1_est.wav was emitted.")
    except Exception as e:
        print(f"[Processing] ⚠️ Dolphin failed or missing: {e}. Falling back to raw audio.")
        wav_to_enroll = tmp_audio

    # 3. ECAPA-TDNN Enrollment
    print("[Processing] Running ECAPA-TDNN Enrollment...")
    try:
        from ecapa_enroll import enroll_from_wav
        emb_path = enroll_from_wav(wav_to_enroll)
        print(f"[Processing] ✅ Enrollment Complete! Saved -> {emb_path}")
        
        # 4. Signal the audio stream to refresh embeddings over the existing socket
        request_hot_reload()
        print("[Processing] 🔁 Requested embedding hot-reload for stream.")
    except Exception as e:
        print(f"[Processing] ❌ Enrollment Failed: {e}")

# =====================================================================
# OpenCV UI Thread (runs synchronously in main loop)
# =====================================================================

def run_camera_ui(camera_index: int | None = None):
    global is_recording, record_start_time, recorded_frames, recorded_audio_chunks
    
    print(f"\n[UI] Opening targeting camera (index={camera_index if camera_index is not None else 'auto'})...")
    with LipTargetingSystem(camera_index=camera_index) as ts:
        
        while True:
            ok, frame = ts._cap.read()
            if not ok:
                break
            
            # This handles MediaPipe tracking silently
            angle, lip_crop, face_crop = ts._process_frame(frame, draw=True)
            
            if is_recording:
                elapsed = time.time() - record_start_time
                cv2.putText(frame, f"REC {elapsed:.1f}s / {MAX_RECORD_SEC}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)
                
                if getattr(ts, 'current_raw_frame', None) is not None:
                    recorded_frames.append(ts.current_raw_frame.copy())
                else:
                    recorded_frames.append(frame.copy())
                
                if elapsed >= MAX_RECORD_SEC:
                    print(f"\n[Recorder] Max duration reached ({MAX_RECORD_SEC}s). Stopping.")
                    is_recording = False
                    ts.is_locked = False
                    
                    # Spawn background processing
                    v_buf = list(recorded_frames)
                    with record_lock:
                        a_buf = list(recorded_audio_chunks)
                    threading.Thread(
                        target=_process_recording_thread, 
                        args=(v_buf, a_buf, elapsed),
                        daemon=True
                    ).start()
            else:
                cv2.putText(frame, "STANDBY - press 'i' to record", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Main App - Camera + Gating", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("i") and not is_recording:
                is_recording = True
                ts.is_locked = True
                record_start_time = time.time()
                recorded_frames.clear()
                with record_lock:
                    recorded_audio_chunks.clear()
                print("\n[Recorder] ● Recording started (5 seconds)...")
                
            elif key == ord("q") or key == 27:
                break

    cv2.destroyAllWindows()


# =====================================================================
# Asyncio Streaming Client
# =====================================================================

async def stream_audio_task(ws_url: str):
    global is_recording, recorded_audio_chunks

    idx, chs = _get_best_mic()
    print(f"🎤 Starting Mic stream (input dev={idx}, output dev=default)...")

    pending_audio: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()
    recv_queue = collections.deque(maxlen=32)

    def audio_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"[Audio] {status}", flush=True)

        # 1. Provide input to Websocket Buffer
        mono = indata[:, 0].copy()
        pending_audio.put(mono)
        
        # 2. Provide input to Recorder if recording
        if is_recording:
            # We record exactly what came in (all channels) before downstreaming
            with record_lock:
                recorded_audio_chunks.append(indata.copy())

        # 3. Read output from Websocket Buffer to Speaker
        if len(recv_queue) > 0:
            while len(recv_queue) > JITTER_BUFFER_SIZE + 1:
                recv_queue.popleft()
            processed = recv_queue.popleft()
            if len(processed) >= frames:
                outdata[:, 0] = processed[:frames]
            else:
                outdata[:frames, 0] = 0.0
                outdata[:len(processed), 0] = processed
        else:
            outdata[:, 0] = 0.0
            stats["drops"] += 1
    while not shutdown_requested.is_set():
        embs, names = load_embeddings()
        print(f"\n🔗 Connecting to {ws_url} …")
        print(f"✓  Starting with {len(names)} speaker(s): {', '.join(names)}")

        playback_started = False
        chunks_buffered = 0
        stats = {"sent": 0, "drops": 0, "sim1": 0.0, "sim2": 0.0, "active": False}
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
                    fresh_embs, fresh_names = load_embeddings()
                    print(f"\n[WebSocket] 🔥 Hot-reloading {len(fresh_names)} speaker(s): {', '.join(fresh_names)}")
                    await ws.send(json.dumps({
                        "type": "init",
                        "embeddings": fresh_embs,
                        "sample_rate": MODEL_SR,
                    }))

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
                    int16_data = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    await ws.send(int16_data.tobytes())
                    stats["sent"] += 1

                if not drained:
                    await asyncio.sleep(0.005)

        async def consumer(ws):
            nonlocal playback_started, chunks_buffered
            HEADER_SIZE = 16
            while not shutdown_requested.is_set():
                raw = await ws.recv()
                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "error":
                            print(f"\n❌  Server error: {msg.get('detail')}")
                    except json.JSONDecodeError:
                        print(f"\n[WebSocket] Ignoring text frame: {raw[:80]}")
                    continue

                if len(raw) > HEADER_SIZE:
                    sim1, sim2, gain1, gain2 = struct.unpack('ffff', raw[:HEADER_SIZE])
                    stats["sim1"], stats["sim2"] = sim1, sim2
                    stats["active"] = (gain1 > 0.5 or gain2 > 0.5)
                    audio_bytes = raw[HEADER_SIZE:]
                else:
                    audio_bytes = raw

                samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                recv_queue.append(samples)

                if not playback_started:
                    chunks_buffered += 1
                    if chunks_buffered >= JITTER_BUFFER_SIZE:
                        playback_started = True

        async def status_printer():
            while not shutdown_requested.is_set():
                await asyncio.sleep(0.3)
                sim_max = max(stats['sim1'], stats['sim2'])
                
                if stats['active']:
                    ind = f"\033[92m🟢 TARGET ({sim_max:.2f})\033[0m"
                elif sim_max > 0.05:
                    ind = f"\033[91m🔴 bg ({sim_max:.2f})\033[0m"
                else:
                    ind = "⚫ sil        "
                    
                sys.stdout.write(f"\r[Gating] {ind} | sent={stats['sent']:4d} drops={stats['drops']:2d}   ")
                sys.stdout.flush()

        try:
            async with websockets.connect(
                ws_url, 
                max_size=2**20, 
                ping_interval=20, 
                ping_timeout=60,
                open_timeout=30  # Give Modal time to boot from cold start
            ) as ws:
                await ws.send(json.dumps({
                    "type": "init",
                    "embeddings": embs,
                    "sample_rate": MODEL_SR,
                }))
                ready_raw = await ws.recv()
                if isinstance(ready_raw, str):
                    resp = json.loads(ready_raw)
                    if resp.get("type") == "error":
                        print(f"❌  Server error: {resp['detail']}")
                        await asyncio.sleep(1.0)
                        continue

                # 1. Use the selected input device (e.g. 'onn. webcam'), 
                # 2. Use the system default output device (e.g. 'MacBook Pro Speakers')
                stream_ctx = sd.Stream(
                    device=(idx, sd.default.device[1]), 
                    samplerate=MODEL_SR,
                    blocksize=CHUNK_SAMPLES,
                    channels=(chs, 1),   # (input_channels, output_channels)
                    dtype="float32",
                    callback=audio_callback,
                )

                with stream_ctx:
                    await asyncio.gather(producer(ws), consumer(ws), status_printer())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if shutdown_requested.is_set():
                break
            print(f"\n[WebSocket] Disconnected: {e}. Reconnecting in 1s...")
            await asyncio.sleep(1.0)

# =====================================================================
# Main Thread Entry
# =====================================================================

def _run_asyncio_thread(url):
    # Runs the websocket loop in a dedicated background thread forever
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(stream_audio_task(url))
    except Exception as e:
        print(f"\n[WebSocket] Thread exited: {e}")
    finally:
        loop.close()

def main():
    parser = argparse.ArgumentParser(description="Unified App")
    parser.add_argument("url", nargs="?", default=os.environ.get("MODAL_WS_URL", ""))
    parser.add_argument("--camera", type=int, default=None, help="Force a specific camera index (e.g., 0)")
    args = parser.parse_args()

    if not args.url:
        print("❌  No WebSocket URL provided.")
        sys.exit(1)

    url = args.url.rstrip("/")
    if not url.endswith("/ws"):
        url += "/ws"

    # Start WebSocket and Audio Streaming in background thread
    t = threading.Thread(target=_run_asyncio_thread, args=(url,), daemon=True)
    t.start()

    # Give it a second to connect before taking over terminal/screen
    time.sleep(1.0)
    
    # Run OpenCV fully synchronously on the main thread
    run_camera_ui(camera_index=args.camera)
    shutdown_requested.set()
    
    t.join(timeout=2.0)
    print("\n👋  Exiting Main App.")

if __name__ == "__main__":
    main()
