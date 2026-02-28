"""
Enroll Target Speaker (Cloud-Matched)
=============================================================

Records 5 seconds of the Target's voice and sends it to the
Modal GPU backend for SepFormer → ECAPA processing. This ensures
the enrollment embedding is in the same domain as runtime embeddings.

Falls back to local ECAPA if no server URL is provided.

Usage
-----
    # Cloud enrollment (recommended — matches runtime pipeline)
    uv run python noise_gate/enroll_target.py --url https://spacial-audio-recognition--spatial-audio-separator-serve.modal.run

    # Local enrollment (fallback)
    uv run python noise_gate/enroll_target.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import requests
import sounddevice as sd
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

DURATION = 5          # seconds
SAMPLE_RATE = 16_000  # 16 kHz mono
CHANNELS = 1


def record_audio(name: str) -> np.ndarray:
    """Record DURATION seconds of audio, return float32 array."""
    input(f"Press Enter, then have {name} speak clearly for {DURATION} seconds … ")
    print(f"🎙  Recording {name} for {DURATION}s …")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print(f"✓  Recording for {name} complete.\n")
    return audio[:, 0]  # (samples,) float32


def enroll_cloud(audio: np.ndarray, server_url: str) -> np.ndarray:
    """Send audio to Modal /enroll endpoint for SepFormer → ECAPA embedding."""
    url = server_url.rstrip("/") + "/enroll"
    print(f"☁️   Sending {len(audio)} samples to {url} …")

    # Convert float32 → int16 for wire
    int16_data = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    resp = requests.post(
        url,
        data=int16_data.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        timeout=120,
    )
    resp.raise_for_status()

    result = resp.json()
    if "error" in result:
        print(f"❌  Server error: {result['error']}")
        sys.exit(1)

    vec = np.array(result["embedding"], dtype=np.float32)
    print(f"✓  Received {vec.shape[0]}-dim SepFormer-domain embedding from cloud")
    return vec


def enroll_local(audio: np.ndarray) -> np.ndarray:
    """Compute ECAPA embedding locally (no SepFormer processing)."""
    from speechbrain.inference.speaker import EncoderClassifier

    print("Loading local ECAPA-TDNN model …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(
            os.path.dirname(__file__), "..", "pretrained_models", "spkrec-ecapa-voxceleb"
        ),
        run_opts={"device": str(device)},
    )
    print(f"✓  Model loaded on {device}\n")

    waveform = torch.from_numpy(audio).unsqueeze(0).float().to(device)
    with torch.no_grad():
        embedding = model.encode_batch(waveform)
        vec = embedding.squeeze().cpu().numpy()

    print(f"✓  Computed {vec.shape[0]}-dim embedding locally (no SepFormer)")
    return vec


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll a target speaker")
    parser.add_argument(
        "--url",
        default=os.environ.get("MODAL_SERVER_URL", ""),
        help="Modal server URL for cloud enrollment (recommended)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Speaker Enrollment")
    print("=" * 60)

    name = input("\nWhat is the name of the user you want to enroll? ").strip()
    if not name:
        name = "Target"

    # Record audio
    audio = record_audio(name)

    # Compute embedding
    if args.url:
        print(f"\n🌐  Using cloud enrollment (SepFormer-domain matched)")
        vec = enroll_cloud(audio, args.url)
    else:
        print(f"\n💻  Using local enrollment (no SepFormer — may cause domain mismatch)")
        print(f"   Tip: use --url to enroll through the cloud pipeline\n")
        vec = enroll_local(audio)

    # Save
    emb_dir = os.path.join(os.path.dirname(__file__), "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    output_path = os.path.join(emb_dir, f"{name}.npy")
    np.save(output_path, vec)

    print(f"\n✓  Saved {vec.shape[0]}-d embedding → {output_path}")
    print(f"   L2 norm: {np.linalg.norm(vec):.4f}")
    print(f"\nEnrollment complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋  Cancelled.")
        sys.exit(0)
