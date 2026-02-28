"""
Bi-State Target Speaker VAD & Spectral Subtraction
====================================================

Streams mic → speakers at native sample rate.

Bi-State Logic:
  - State A (Target X): Spectral subtraction heavily removes background, gain boosts X.
  - State B (Background): Uses a small amount of spectral subtraction to suppress background slightly, or passthrough naturally. Updates noise profile.

Usage
-----
    uv run python noise_gate/enroll_target.py    # record target.npy first
    uv run python noise_gate/target_speaker_vad.py # then run the gate
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time

import numpy as np
import noisereduce as nr
import sounddevice as sd
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from speechbrain.inference.speaker import EncoderClassifier

# =====================================================================
# Configuration
# =====================================================================

BLOCK_SIZE = 512           # Audio callback frame size (≈ 10 ms at 48 kHz)
FFT_SIZE = 1024            # STFT window length (≈ 21 ms at 48 kHz)
HOP_SIZE = BLOCK_SIZE      # STFT hop = callback frame size (50 % overlap)

BUFFER_DURATION = 0.6      # Ring-buffer length for AI thread (600ms trailing)
CHANNELS = 1               # Mono
MODEL_SR = 16_000          # SpeechBrain expected sample rate

# Thresholds
THRESHOLD = 0.25           # Similarity threshold to trigger state change
RMS_THRESHOLD = 5e-4       # Minimum loudness to bother running AI model

# Spectral processing
TARGET_GAIN = 2.5          # Amplification for Target X
OVERSUBTRACT_TARGET = 3.0  # Aggressive noise removal when Target X is speaking
SPECTRAL_FLOOR = 0.05      # Minimum fraction of original magnitude kept

OVERSUBTRACT_BG = 0.0      # NO noise suppression when nobody is speaking (normal background)
BG_GAIN = 1.0              # Normal listening volume for background

NOISE_EMA = 0.95           # EMA factor for noise estimate (background tracking)

# Gain smoothing (prevents audible clicks on state transitions)
TRANSITION_ATTACK = 0.05   # Seconds to transition states
TRANSITION_RELEASE = 0.25

TARGET_PATH = os.path.join(os.path.dirname(__file__), "target.npy")

# =====================================================================
# Global Shared State
# =====================================================================

target_is_speaking: bool = False
current_sim_t: float = 0.0

# --- ring buffer for AI thread ---------------------------------------
ring_buffer: np.ndarray | None = None
ring_write_pos: int = 0
ring_lock = threading.Lock()

# --- spectral processor state ----------------------------------------
noise_spectrum: np.ndarray | None = None   # (FFT_SIZE//2+1,) running avg
noise_ready: bool = False

# --- frame queue -----------------------------------------------------
input_frames: collections.deque = collections.deque(maxlen=64)
output_frames: collections.deque = collections.deque(maxlen=64)

native_sr: int = 0
window: np.ndarray | None = None

# =====================================================================
# 1. Audio Callback
# =====================================================================

def audio_callback(indata, outdata, frames, time_info, status):
    global ring_write_pos

    if status:
        print(f"  ⚠ {status}", file=sys.stderr)

    mono = indata[:, 0].copy()

    with ring_lock:
        buf_len = len(ring_buffer)
        end = ring_write_pos + len(mono)
        if end <= buf_len:
            ring_buffer[ring_write_pos:end] = mono
        else:
            first = buf_len - ring_write_pos
            ring_buffer[ring_write_pos:] = mono[:first]
            ring_buffer[: end - buf_len] = mono[first:]
        ring_write_pos = end % buf_len

    input_frames.append(mono)

    try:
        processed = output_frames.popleft()
        outdata[:, 0] = processed
    except IndexError:
        outdata[:, 0] = mono  # fallback to regular pass-through if slow


# =====================================================================
# 2. Spectral Processing Thread
# =====================================================================

def spectral_thread():
    global noise_spectrum, noise_ready

    prev_input = np.zeros(FFT_SIZE - HOP_SIZE, dtype=np.float32)
    overlap_tail = np.zeros(FFT_SIZE - HOP_SIZE, dtype=np.float32)

    smooth_target = 0.0
    
    attack_coeff = 1.0 - np.exp(-1.0 / (TRANSITION_ATTACK * native_sr / HOP_SIZE))
    release_coeff = 1.0 - np.exp(-1.0 / (TRANSITION_RELEASE * native_sr / HOP_SIZE))

    while True:
        try:
            frame = input_frames.popleft()
        except IndexError:
            time.sleep(0.001)
            continue

        analysis = np.concatenate([prev_input, frame])
        prev_input = analysis[HOP_SIZE:].copy()

        windowed = analysis * window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # ── Smooth state transitions ────────────────────────────────
        t_trg = 1.0 if target_is_speaking else 0.0
        smooth_target += (attack_coeff if t_trg > smooth_target else release_coeff) * (t_trg - smooth_target)
        smooth_bg = 1.0 - smooth_target

        # ── Spectral Modification ───────────────────────────────────
        if noise_spectrum is None:
            noise_spectrum = magnitude.copy()
            noise_ready = True

        if noise_ready:
            # Update ambient noise profile only when background is fully active
            if smooth_target < 0.1:
                noise_spectrum = NOISE_EMA * noise_spectrum + (1.0 - NOISE_EMA) * magnitude

            # 1. Target output: aggressively subtract noise, amplify target voice, and CRUSH the residual background floor
            clean_target = np.maximum(magnitude - OVERSUBTRACT_TARGET * noise_spectrum, SPECTRAL_FLOOR * magnitude)
            target_mag = np.where(clean_target <= SPECTRAL_FLOOR * magnitude * 2.0, clean_target * 0.01, clean_target * TARGET_GAIN)
            
            # 2. Background output: softly subtract noise to reduce hum, but preserve volume
            bg_mag = np.maximum(magnitude - OVERSUBTRACT_BG * noise_spectrum, SPECTRAL_FLOOR * magnitude) * BG_GAIN

            clean_mag = (smooth_target * target_mag) + (smooth_bg * bg_mag)
        else:
            clean_mag = magnitude

        # ── Synthesis & Overlap-Add ─────────────────────────────────
        clean_spectrum = clean_mag * np.exp(1j * phase)
        clean_frame = np.fft.irfft(clean_spectrum, n=FFT_SIZE).astype(np.float32)
        clean_frame *= window

        output_block = clean_frame[:HOP_SIZE] + overlap_tail
        overlap_tail = clean_frame[HOP_SIZE:].copy()

        output_frames.append(output_block)


# =====================================================================
# 3. AI Inference Thread
# =====================================================================

def inference_thread(model: EncoderClassifier, target_emb: torch.Tensor, device: torch.device):
    global target_is_speaking, current_sim_t

    resampler = torchaudio.transforms.Resample(native_sr, MODEL_SR).to(device)
    print("\n🎧  AI inference thread running …\n")

    while True:
        with ring_lock:
            snapshot = ring_buffer.copy()

        rms = np.sqrt(np.mean(snapshot ** 2))
        
        # Fast VAD: if quiet, skip deep learning altogether (saves CPU and latency)
        if rms < RMS_THRESHOLD:
            target_is_speaking = False
            current_sim_t = 0.0
            print(f"\r   BACKGROUND (quiet)   rms={rms:.5f}                 ", end="", flush=True)
            time.sleep(0.05)
            continue

        waveform = torch.from_numpy(snapshot).unsqueeze(0).float().to(device)
        waveform_16k = resampler(waveform)

        # Denoise buffer chunk to improve ECAPA-TDNN robustness in noise
        audio_16k = waveform_16k.squeeze().cpu().numpy()
        audio_clean = nr.reduce_noise(y=audio_16k, sr=MODEL_SR, stationary=True, prop_decrease=0.9)
        waveform_clean = torch.from_numpy(audio_clean).unsqueeze(0).float().to(device)

        with torch.no_grad():
            emb = model.encode_batch(waveform_clean).squeeze()

        sim_t = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        current_sim_t = sim_t
        
        # Decision Logic
        if sim_t > THRESHOLD:
            target_is_speaking = True
            tag = "🟢 TARGET X   "
        else:
            target_is_speaking = False
            tag = "   BACKGROUND "

        print(f"\r{tag} sim_T={sim_t:+.2f} rms={rms:.4f}  ", end="", flush=True)
        time.sleep(0.05)


# =====================================================================
# Main
# =====================================================================

def main():
    global ring_buffer, native_sr, window

    if not os.path.exists(TARGET_PATH):
        print(f"❌  target.npy not found!")
        print("   Run:  uv run python noise_gate/enroll_target.py")
        sys.exit(1)

    target_np = np.load(TARGET_PATH)
    print(f"✓  Target X embedding loaded ({target_np.shape[0]}-d)")

    dev_info = sd.query_devices(kind="input")
    native_sr = int(dev_info["default_samplerate"])
    print(f"✓  Mic: {native_sr} Hz")

    ring_buffer = np.zeros(int(native_sr * BUFFER_DURATION), dtype=np.float32)
    window = np.hanning(FFT_SIZE).astype(np.float32)
    print(f"✓  FFT: {FFT_SIZE} samples, hop: {HOP_SIZE}")

    print("   Loading ECAPA-TDNN …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": str(device)},
    )
    print(f"✓  Model loaded on {device}")

    target_emb = torch.from_numpy(target_np).float().to(device)

    t_spec = threading.Thread(target=spectral_thread, daemon=True)
    t_spec.start()

    t_ai = threading.Thread(target=inference_thread, args=(model, target_emb, device), daemon=True)
    t_ai.start()

    print(f"\n🎤  Streaming — Threshold={THRESHOLD}, Ctrl-C to stop\n")

    with sd.Stream(samplerate=native_sr, blocksize=BLOCK_SIZE, channels=CHANNELS, dtype="float32", callback=audio_callback):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n👋  Stopped.")


if __name__ == "__main__":
    main()
