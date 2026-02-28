"""
Cloud Noise Gate — Local Mac Client
====================================

Streams microphone audio to the Modal GPU backend via WebSocket
for real-time source separation and biometric speaker gating.

The Modal backend runs SepFormer (separation) + ECAPA-TDNN (verification)
on an H100 GPU and returns gated audio: enrolled speakers at full volume,
unenrolled speakers dimmed to 5% (−26 dB).

Usage
-----
    # First deploy the backend
    modal deploy modal_separator.py

    # Then run the client (paste your Modal endpoint URL)
    uv run python cloud_noise_gate.py

    # Quick connectivity test
    uv run python cloud_noise_gate.py --test
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import struct
import sys
import time

import numpy as np
import sounddevice as sd

try:
    import websockets
except ImportError:
    print("❌  Missing dependency: pip install websockets")
    sys.exit(1)

# =====================================================================
# Configuration
# =====================================================================

MODEL_SR = 16_000                               # Must match Modal backend
CHUNK_DURATION = 0.15                            # 150 ms per chunk
CHUNK_SAMPLES = int(MODEL_SR * CHUNK_DURATION)   # 2400 samples
CHANNELS = 1

# Jitter buffer: hold N chunks before starting playback
JITTER_BUFFER_SIZE = 2

# WebSocket endpoint — set via env var or CLI arg
DEFAULT_WS_URL = os.environ.get("MODAL_WS_URL", "")

EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "noise_gate", "embeddings")

# =====================================================================
# Load enrolled speaker embeddings
# =====================================================================

def load_embeddings() -> tuple[list[list[float]], list[str]]:
    """Load all *.npy embeddings from the embeddings directory."""
    if not os.path.exists(EMBEDDINGS_DIR):
        print(f"❌  Embeddings directory not found: {EMBEDDINGS_DIR}")
        print("   Run:  uv run python noise_gate/enroll_target.py")
        sys.exit(1)

    npy_files = sorted(f for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".npy"))
    if not npy_files:
        print(f"❌  No .npy files in {EMBEDDINGS_DIR}")
        print("   Run:  uv run python noise_gate/enroll_target.py")
        sys.exit(1)

    embeddings = []
    names = []
    for f in npy_files:
        vec = np.load(os.path.join(EMBEDDINGS_DIR, f))
        embeddings.append(vec.tolist())
        names.append(f[:-4])

    return embeddings, names


# =====================================================================
# Async streaming client
# =====================================================================

async def stream(ws_url: str):
    """Main streaming loop: mic → WebSocket → playback."""

    # 1. Load embeddings
    embeddings, names = load_embeddings()
    print(f"✓  Loaded embeddings for {len(names)} speaker(s): {', '.join(names)}")

    # 2. Shared state between audio callback and async tasks
    send_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
    recv_queue: collections.deque[np.ndarray] = collections.deque(maxlen=32)
    playback_started = False
    chunks_buffered = 0
    stats = {"rtt_ms": 0.0, "chunks_sent": 0, "underruns": 0,
             "sim1": 0.0, "sim2": 0.0, "gain1": 0.0, "gain2": 0.0,
             "target_active": False}

    # Accumulation buffer for the audio callback
    accum_buf = np.zeros(0, dtype=np.float32)
    accum_lock = asyncio.Lock()

    # Event loop reference for thread-safe scheduling
    loop = asyncio.get_event_loop()

    # ── Audio callback (runs in sounddevice's C thread) ──────────
    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal playback_started, chunks_buffered, accum_buf

        if status:
            print(f"  ⚠ {status}", file=sys.stderr)

        # Capture input: accumulate samples
        mono = indata[:, 0].copy()
        # We need to accumulate in a thread-safe way
        # Use a simple approach: append to a list that the producer drains
        audio_callback._pending.append(mono)

        # Output: pull from receive queue
        if len(recv_queue) > 0:
            # If network jitter caused a burst of chunks, drop old ones to catch up to real-time
            # This prevents a temporary network stutter from causing a persistent latency increase
            while len(recv_queue) > JITTER_BUFFER_SIZE + 1:
                recv_queue.popleft()
                # You might log drops here if needed, but keeping it simple for now
                
            processed = recv_queue.popleft()
            # Ensure correct length
            if len(processed) >= frames:
                outdata[:, 0] = processed[:frames]
            else:
                outdata[:frames, 0] = 0.0
                outdata[:len(processed), 0] = processed
        else:
            # Underrun: output silence
            outdata[:, 0] = 0.0
            stats["underruns"] += 1

    audio_callback._pending = []

    # ── Producer: mic → WebSocket ────────────────────────────────
    async def producer(ws):
        nonlocal accum_buf
        chunk_bytes_target = CHUNK_SAMPLES  # 2400 samples per chunk

        while True:
            # Drain pending audio from callback
            if audio_callback._pending:
                new_data = audio_callback._pending.copy()
                audio_callback._pending.clear()
                new_samples = np.concatenate(new_data)
                accum_buf = np.concatenate([accum_buf, new_samples])

            # Send full chunks
            while len(accum_buf) >= chunk_bytes_target:
                chunk = accum_buf[:chunk_bytes_target]
                accum_buf = accum_buf[chunk_bytes_target:]

                # Convert float32 → int16 for wire
                int16_data = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                await ws.send(int16_data.tobytes())
                stats["chunks_sent"] += 1

            await asyncio.sleep(0.005)  # 5ms poll

    # ── Consumer: WebSocket → playback queue ─────────────────────
    async def consumer(ws):
        nonlocal playback_started, chunks_buffered
        HEADER_SIZE = 16  # 4 floats × 4 bytes

        while True:
            raw = await ws.recv()
            t_recv = time.perf_counter()

            # Parse 16-byte gating header: sim1, sim2, gain1, gain2
            if len(raw) > HEADER_SIZE:
                sim1, sim2, gain1, gain2 = struct.unpack('ffff', raw[:HEADER_SIZE])
                audio_bytes = raw[HEADER_SIZE:]
                stats["sim1"] = sim1
                stats["sim2"] = sim2
                stats["gain1"] = gain1
                stats["gain2"] = gain2
                stats["target_active"] = (gain1 > 0.5 or gain2 > 0.5)
            else:
                audio_bytes = raw

            # Decode int16 → float32
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            recv_queue.append(samples)

            # Wait for jitter buffer to fill before we start consuming
            if not playback_started:
                chunks_buffered += 1
                if chunks_buffered >= JITTER_BUFFER_SIZE:
                    playback_started = True
                    print("🔊  Playback started (jitter buffer filled)")

    # ── Status printer ───────────────────────────────────────────
    async def status_printer():
        while True:
            await asyncio.sleep(0.3)
            sim_max = max(stats['sim1'], stats['sim2'])
            
            if stats['target_active']:
                indicator = f"\033[92m🟢 TARGET SPEAKING (sim={sim_max:.2f})\033[0m"
            elif sim_max > 0.05:
                indicator = f"\033[91m🔴 background (sim={sim_max:.2f})\033[0m"
            else:
                indicator = "⚫ silence"
            print(
                f"\r  {indicator}  │  "
                f"sent={stats['chunks_sent']:5d}  "
                f"buf={len(recv_queue):2d}  "
                f"drops={stats['underruns']}   ",
                end="", flush=True,
            )

    # ── 3. Connect and run ───────────────────────────────────────
    print(f"\n🔗  Connecting to {ws_url} …")

    async with websockets.connect(
        ws_url,
        max_size=2**20,           # 1 MB max message
        ping_interval=20,
        ping_timeout=60,
        close_timeout=5,
    ) as ws:
        # Handshake
        init_msg = json.dumps({
            "type": "init",
            "embeddings": embeddings,
            "sample_rate": MODEL_SR,
        })
        await ws.send(init_msg)

        resp_raw = await ws.recv()
        resp = json.loads(resp_raw)
        if resp.get("type") == "error":
            print(f"❌  Server error: {resp['detail']}")
            return
        print(f"✓  Server ready — {resp.get('speakers', '?')} speaker(s) enrolled")

        # Start audio stream
        print(f"🎤  Starting audio stream at {MODEL_SR} Hz …")
        print("   Press Ctrl-C to stop\n")

        stream_ctx = sd.Stream(
            samplerate=MODEL_SR,
            blocksize=CHUNK_SAMPLES,
            channels=CHANNELS,
            dtype="float32",
            callback=audio_callback,
        )

        with stream_ctx:
            # Run producer, consumer, and status printer concurrently
            try:
                await asyncio.gather(
                    producer(ws),
                    consumer(ws),
                    status_printer(),
                )
            except KeyboardInterrupt:
                pass

    print("\n\n👋  Disconnected.")


# =====================================================================
# Quick connectivity test
# =====================================================================

async def test_connection(ws_url: str):
    """Send one chunk of silence and verify the round-trip."""
    embeddings, names = load_embeddings()
    print(f"✓  Loaded embeddings for {len(names)} speaker(s): {', '.join(names)}")
    print(f"🔗  Connecting to {ws_url} …")

    async with websockets.connect(ws_url, max_size=2**20) as ws:
        # Handshake
        await ws.send(json.dumps({
            "type": "init",
            "embeddings": embeddings,
            "sample_rate": MODEL_SR,
        }))
        resp = json.loads(await ws.recv())
        print(f"✓  Handshake: {resp}")

        # Send 10 chunks of silence, measure RTT
        silence = np.zeros(CHUNK_SAMPLES, dtype=np.int16).tobytes()
        rtts = []

        for i in range(10):
            t0 = time.perf_counter()
            await ws.send(silence)
            reply = await ws.recv()
            rtt = (time.perf_counter() - t0) * 1000
            rtts.append(rtt)

            reply_samples = len(reply) // 2  # int16 = 2 bytes per sample
            print(f"  chunk {i+1:2d}: {reply_samples} samples, RTT={rtt:.1f}ms")

        avg_rtt = sum(rtts) / len(rtts)
        print(f"\n✓  Test passed — avg RTT: {avg_rtt:.1f}ms")
        print(f"   Expected playback latency: ~{avg_rtt + CHUNK_DURATION*1000:.0f}ms "
              f"(RTT + {CHUNK_DURATION*1000:.0f}ms chunk)")

    print("👋  Disconnected.")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cloud Noise Gate — stream mic audio to Modal for GPU separation"
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=DEFAULT_WS_URL,
        help="WebSocket URL of the Modal endpoint (wss://...)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick connectivity test instead of full streaming",
    )
    args = parser.parse_args()

    if not args.url:
        print("❌  No WebSocket URL provided.")
        print("   Usage:  uv run python cloud_noise_gate.py wss://your-modal-url/ws")
        print("   Or set MODAL_WS_URL environment variable.")
        sys.exit(1)

    # Normalize URL: ensure it ends with /ws
    url = args.url.rstrip("/")
    if not url.endswith("/ws"):
        url += "/ws"

    print("=" * 60)
    print("  Cloud Noise Gate — GPU Source Separation")
    print("=" * 60)

    if args.test:
        asyncio.run(test_connection(url))
    else:
        asyncio.run(stream(url))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋  Stopped.")
        sys.exit(0)
