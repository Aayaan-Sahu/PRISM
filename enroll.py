"""
Speaker Enrollment via Beamforming
===================================

Records directional audio using the 2-mic beamforming setup from
main.py, trims silence with Silero VAD, and computes a stable
speaker voiceprint (192-d ECAPA-TDNN embedding).

Usage
-----
    python enroll.py --name aayaan --angle -45 --duration 7

The resulting voiceprint is saved to ``voiceprints/<name>.pt``.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from config import (
    AGG_DEVICE_INDEX,
    CHANNELS,
    ENROLL_DURATION_SEC,
    MIC_DISTANCE_M,
    SAMPLE_RATE,
    SPEED_OF_SOUND,
    TARGET_SR,
    VOICEPRINT_DIR,
)
from embeddings import SpeakerEncoder
from vad import VoiceActivityDetector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)


# ─── Beamforming (copied from main.py, not imported to avoid side effects) ───
def _beamform(mic_a: np.ndarray, mic_b: np.ndarray, steer_angle_deg: float, fs: int) -> np.ndarray:
    """Delay-and-sum beamformer steered toward *steer_angle_deg*.

    Returns mono float32 array.
    """
    tau_steer = MIC_DISTANCE_M * np.sin(np.radians(steer_angle_deg)) / SPEED_OF_SOUND

    n_fft = 1
    while n_fft < len(mic_a):
        n_fft <<= 1

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * tau_steer)

    B = np.fft.rfft(mic_b, n=n_fft)
    mic_b_delayed = np.fft.irfft(B * phase_shift, n=n_fft)[: len(mic_a)]

    output = 0.5 * (mic_a + mic_b_delayed)
    return output.astype(np.float32)


# ─── Multi-chunk embedding ───────────────────────────────────────
def encode_chunks(
    encoder: SpeakerEncoder,
    waveform: torch.Tensor,
    sr: int,
    chunk_sec: float = 3.0,
    hop_sec: float = 1.0,
) -> torch.Tensor:
    """Split waveform into overlapping chunks, encode each, and average.

    Parameters
    ----------
    encoder : SpeakerEncoder
    waveform : torch.Tensor
        1-D float tensor at *sr* Hz.
    sr : int
    chunk_sec : float
        Length of each chunk in seconds.
    hop_sec : float
        Hop between chunks in seconds.

    Returns
    -------
    torch.Tensor
        Averaged 192-d speaker embedding.
    """
    chunk_samples = int(chunk_sec * sr)
    hop_samples = int(hop_sec * sr)

    if len(waveform) < chunk_samples:
        # Audio shorter than one chunk — encode the whole thing
        logger.warning(
            "Audio (%.1fs) shorter than chunk size (%.1fs) — encoding as-is",
            len(waveform) / sr,
            chunk_sec,
        )
        return encoder.encode(waveform)

    embeddings = []
    start = 0
    while start + chunk_samples <= len(waveform):
        chunk = waveform[start : start + chunk_samples]
        emb = encoder.encode(chunk)
        embeddings.append(emb)
        start += hop_samples

    stacked = torch.stack(embeddings)  # [N, 192]
    averaged = stacked.mean(dim=0)     # [192]

    # L2-normalise for consistent cosine similarity
    averaged = averaged / averaged.norm()

    logger.info(
        "Encoded %d chunks (%.1fs each, %.1fs hop) → averaged embedding",
        len(embeddings),
        chunk_sec,
        hop_sec,
    )
    return averaged


# ─── Main enrollment flow ────────────────────────────────────────
def enroll(
    name: str,
    steer_angle_deg: float,
    duration_sec: float = ENROLL_DURATION_SEC,
    device_index: int = AGG_DEVICE_INDEX,
) -> torch.Tensor:
    """Record, beamform, VAD-trim, and save a speaker voiceprint.

    Parameters
    ----------
    name : str
        Speaker name (used as filename).
    steer_angle_deg : float
        Beamformer steering angle in degrees.
    duration_sec : float
        Recording duration.
    device_index : int
        Audio device index.

    Returns
    -------
    torch.Tensor
        The 192-d voiceprint.
    """
    info = sd.query_devices(device_index, "input")
    fs = int(info["default_samplerate"])
    n_samples = int(fs * duration_sec)

    print(f"\n🎙  Recording {duration_sec}s from device {device_index} @ {fs} Hz")
    print(f"   Beamforming toward {steer_angle_deg:+.1f}°")
    print(f"   Speak now …\n")

    raw = sd.rec(
        n_samples,
        samplerate=fs,
        channels=CHANNELS,
        device=device_index,
        dtype="float32",
    )
    sd.wait()
    print("   ✅ Recording complete\n")

    # Downmix to 2 mics (each Yeti's L+R averaged)
    mic_a = 0.5 * (raw[:, 0] + raw[:, 1])
    mic_b = 0.5 * (raw[:, 2] + raw[:, 3])

    # Beamform toward the target angle
    beamformed = _beamform(mic_a, mic_b, steer_angle_deg, fs)
    logger.info("Beamformed %d samples toward %.1f°", len(beamformed), steer_angle_deg)

    # Convert to torch tensor and resample to 16 kHz for ECAPA
    waveform = torch.from_numpy(beamformed).float()
    if fs != TARGET_SR:
        waveform = torchaudio.transforms.Resample(fs, TARGET_SR)(waveform)
        logger.info("Resampled %d Hz → %d Hz", fs, TARGET_SR)

    # VAD: trim silence
    vad = VoiceActivityDetector()
    waveform = vad.trim_silence(waveform, TARGET_SR)

    if len(waveform) < TARGET_SR:  # less than 1 second of speech
        print("⚠️  Less than 1 second of speech detected. Try again in a quieter spot.")
        sys.exit(1)

    # Encode
    encoder = SpeakerEncoder()
    voiceprint = encode_chunks(encoder, waveform, TARGET_SR)

    # Save
    VOICEPRINT_DIR.mkdir(exist_ok=True)
    save_path = VOICEPRINT_DIR / f"{name}.pt"
    torch.save(voiceprint, save_path)

    print(f"💾 Voiceprint saved → {save_path}")
    print(f"   Shape: {tuple(voiceprint.shape)}")
    print(f"   Norm:  {voiceprint.norm():.4f}")

    return voiceprint


# ─── CLI ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll a speaker using beamformed directional audio."
    )
    parser.add_argument(
        "--name", "-n",
        required=True,
        help="Speaker name (used as voiceprint filename)",
    )
    parser.add_argument(
        "--angle", "-a",
        type=float,
        required=True,
        help="Beamformer steering angle in degrees",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=ENROLL_DURATION_SEC,
        help=f"Recording duration in seconds (default: {ENROLL_DURATION_SEC})",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=AGG_DEVICE_INDEX,
        help=f"Audio device index (default: {AGG_DEVICE_INDEX})",
    )

    args = parser.parse_args()
    enroll(args.name, args.angle, args.duration, args.device)


if __name__ == "__main__":
    main()
