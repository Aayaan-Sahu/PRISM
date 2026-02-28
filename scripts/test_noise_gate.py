"""
End-to-End Bi-State Noise Gate Test
=====================================

Offline test that demonstrates the speaker-gated spectral noise filter.

Steps:
  1. Load enrolled target.npy
  2. Record 15 s of noisy audio (alternate TARGET X and NOISE)
  3. STFT-based spectral subtraction (aggressive for X, mild for background)
  4. Save original + filtered .wav for A/B comparison

Usage
-----
    uv run python scripts/test_noise_gate.py
"""

from __future__ import annotations

import collections
import os
import sys
import time
import wave as wave_mod

import numpy as np
import noisereduce as nr
import sounddevice as sd
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from speechbrain.inference.speaker import EncoderClassifier

# ── Config ───────────────────────────────────────────────────────────
RECORD_DURATION = 15
SAMPLE_RATE = 16_000       # Record at model rate for simplicity
CHANNELS = 1

# Speaker-ID gating
CHUNK_DURATION = 0.34      # trailing window
RECENT_DURATION = 0.15     # Short chunk for interrupt detection
CHUNK_HOP = 0.05           # AI analysis hop (seconds)
THRESHOLD = 0.28           # Higher threshold prevents false positives (e.g. your friend)
HANG_THRESHOLD = 0.25      # Lower threshold to KEEP gate open if it was already open
HANG_TIME = 0.8            # Seconds to keep gate open after Target X stops speaking

# Spectral processing
FFT_SIZE = 1024
HOP_SIZE = 512

TARGET_GAIN = 2.5          # Amplification for Target X
OVERSUBTRACT_TARGET = 1.5  # Heavy penalty against Background Y's profile
OVERSUBTRACT_BG = 0.0      # NO noise removal when nobody speaks (normal background)
BG_GAIN = 1.0              # Normal listening volume for background
SPECTRAL_FLOOR = 0.05      # 5% safety net to prevent muffling Target X's harmonics

NOISE_EMA = 0.95           # Exponential moving average for noise estimate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "noise_gate", "embeddings")

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Save float32 numpy → 16-bit WAV."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave_mod.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# =====================================================================
# Spectral Processing (offline, chunk-by-chunk simulate)
# =====================================================================

def spectral_process(
    audio: np.ndarray,
    state_map: np.ndarray,
) -> np.ndarray:
    """
    STFT-based spectral processing guided by a per-sample state map.
    """
    window = np.hanning(FFT_SIZE).astype(np.float32)
    output = np.zeros_like(audio)
    norm = np.zeros_like(audio)

    n_frames = (len(audio) - FFT_SIZE) // HOP_SIZE + 1
    noise_est = None

    smooth_target = 0.0
    
    # Approx 50ms attack, 250ms release
    attack_coeff = 1.0 - np.exp(-1.0 / (0.05 * SAMPLE_RATE / HOP_SIZE))
    release_coeff = 1.0 - np.exp(-1.0 / (0.25 * SAMPLE_RATE / HOP_SIZE))

    for i in range(n_frames):
        start = i * HOP_SIZE
        frame = audio[start: start + FFT_SIZE] * window

        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Majority vote for state in this window
        window_states = state_map[start: start + FFT_SIZE]
        tgt_count = np.sum(window_states == 1)
        t_trg = 1.0 if tgt_count > FFT_SIZE // 2 else 0.0

        smooth_target += (attack_coeff if t_trg > smooth_target else release_coeff) * (t_trg - smooth_target)
        smooth_bg = 1.0 - smooth_target

        if noise_est is None:
            noise_est = magnitude.copy()

        if smooth_target < 0.1:
            noise_est = NOISE_EMA * noise_est + (1 - NOISE_EMA) * magnitude

        # Target output (Only allow voice through, drastically duck background)
        # We subtract noise, boost the voice, but then we must also pull down any residual noise floor
        clean_target = np.maximum(magnitude - OVERSUBTRACT_TARGET * noise_est, SPECTRAL_FLOOR * magnitude)
        
        # We apply TARGET_GAIN, but if a bin is close to SPECTRAL_FLOOR (i.e. it's just noise), we crush it to near 0.
        target_mag = np.where(clean_target <= SPECTRAL_FLOOR * magnitude * 2.0, clean_target * 0.01, clean_target * TARGET_GAIN)
        
        # Background
        bg_mag = np.maximum(magnitude - OVERSUBTRACT_BG * noise_est, SPECTRAL_FLOOR * magnitude) * BG_GAIN

        clean_mag = (smooth_target * target_mag) + (smooth_bg * bg_mag)

        # Reconstruct
        clean = np.fft.irfft(clean_mag * np.exp(1j * phase), n=FFT_SIZE)
        clean = clean.astype(np.float32) * window

        output[start: start + FFT_SIZE] += clean
        norm[start: start + FFT_SIZE] += window ** 2

    norm = np.maximum(norm, 1e-8)
    return output / norm


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("  Bi-State Spectral Noise Filter — End-to-End Test")
    print("=" * 60)

    print("\n  Loading ECAPA-TDNN …")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(os.path.dirname(__file__), "..",
                             "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": str(device)},
    )
    print(f"  ✓  Model on {device}")

    if not os.path.exists(EMBEDDINGS_DIR) or not any(f.endswith(".npy") for f in os.listdir(EMBEDDINGS_DIR)):
        print(f"\n❌  No target embeddings found in {EMBEDDINGS_DIR}!")
        print("   Run:  uv run python noise_gate/enroll_target.py first")
        sys.exit(1)

    target_list = []
    target_names = []
    for f in sorted(os.listdir(EMBEDDINGS_DIR)):
        if f.endswith(".npy"):
            target_list.append(np.load(os.path.join(EMBEDDINGS_DIR, f)))
            target_names.append(f[:-4])
            
    target_np = np.stack(target_list)
    target_emb = torch.from_numpy(target_np).float().to(device)
    print(f"  ✓  Loaded target embeddings for {len(target_list)} users: {', '.join(target_names)}")

    print(f"\n── STEP 1 & 2: Record {RECORD_DURATION}s and Live Classify ──")
    print("  TIP: alternate between NOISE and TARGET X.")
    input(f"\n  Press Enter to start … ")

    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    hop_samples = int(CHUNK_HOP * SAMPLE_RATE)
    total_hops = int(RECORD_DURATION * SAMPLE_RATE / hop_samples)
    
    audio_full = []
    state_map_list = []
    
    q = collections.deque()
    def audio_callback(indata, frames, t_info, status):
        q.append(indata[:, 0].copy())
        
    rolling_buffer = np.zeros(chunk_samples, dtype=np.float32)
    last_target_time = -999.0
    
    print("  🎙  Recording ...")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=hop_samples, callback=audio_callback):
        state = 0
        for hop_idx in range(total_hops):
            while len(q) == 0:
                time.sleep(0.005)
            hop_data = q.popleft()
            audio_full.append(hop_data)
            
            rolling_buffer = np.roll(rolling_buffer, -hop_samples)
            rolling_buffer[-hop_samples:] = hop_data
            
            chunk = rolling_buffer
            rms = np.sqrt(np.mean(chunk ** 2))

            t_sec = (hop_idx * hop_samples) / SAMPLE_RATE

            if rms < 5e-4:
                state = 0
                tag = "   BACKGROUND "
                infer_time_ms = 0.0
                sim_t = 0.0
            else:
                chunk_clean = nr.reduce_noise(y=chunk, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.9)
                ct_full = torch.from_numpy(chunk_clean).unsqueeze(0).float().to(device)
                
                recent_samples = int(RECENT_DURATION * SAMPLE_RATE)
                ct_recent = ct_full[:, -recent_samples:]
                
                pad_len = ct_full.shape[1] - ct_recent.shape[1]
                ct_recent_padded = torch.nn.functional.pad(ct_recent, (pad_len, 0))
                
                batch = torch.cat([ct_full, ct_recent_padded], dim=0)
                
                t0 = time.perf_counter()
                with torch.no_grad():
                    ce = model.encode_batch(batch).squeeze()
                    
                # Compare both audio embeddings to all stored Target embeddings
                # ce shape is [2, 192] -> unsqueeze to [2, 1, 192]
                # target_emb is [N, 192] -> unsqueeze to [1, N, 192]
                # output is [2, N]
                sims = torch.nn.functional.cosine_similarity(ce.unsqueeze(1), target_emb.unsqueeze(0), dim=-1)
                sim_t = sims.max().item()
                infer_time_ms = (time.perf_counter() - t0) * 1000.0
                
                # Logic with Hysteresis & Dual Thresholds
                if sim_t > THRESHOLD:
                    # Strong match
                    state = 1
                    tag = "🟢 TARGET X   "
                    last_target_time = t_sec
                elif state == 1 and sim_t > HANG_THRESHOLD:
                    # Medium match while already open
                    state = 1
                    tag = "🟢 TARGET X   "
                    last_target_time = t_sec
                else:
                    # Weak match: hang time decay
                    if t_sec - last_target_time < HANG_TIME:
                        state = 1
                        tag = "🟢 TARGET X   "
                    else:
                        state = 0
                        tag = "   BACKGROUND "

            state_map_list.append(np.full(hop_samples, state, dtype=np.int32))

            if rms < 5e-4:
                print(f"\r  t={t_sec:5.1f}s  {tag} (rms={rms:.5f} < 5e-4)                  ", end="", flush=True)
            else:
                print(f"\r  t={t_sec:5.1f}s  {tag} (rms={rms:.4f} sim_T={sim_t:+.2f} infer={infer_time_ms:4.1f}ms)", end="", flush=True)
                
    noisy = np.concatenate(audio_full)
    state_map = np.concatenate(state_map_list)

    print("\n\n── STEP 3: Spectral Processing ──")
    filtered = spectral_process(noisy, state_map)

    orig_path = os.path.abspath(os.path.join(OUTPUT_DIR, "output_original.wav"))
    filt_path = os.path.abspath(os.path.join(OUTPUT_DIR, "output_filtered.wav"))
    save_wav(orig_path, noisy, SAMPLE_RATE)
    save_wav(filt_path, filtered, SAMPLE_RATE)

    print(f"\n── DONE ──")
    print(f"  Original : {orig_path}")
    print(f"  Filtered : {filt_path}")
    print(f"\n  Compare:  open {orig_path}")
    print(f"            open {filt_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋  Cancelled.")
        sys.exit(0)
