"""
Enroll Target Speaker
=============================================================

Records 5 seconds of the Target's voice and saves the ECAPA-TDNN speaker embedding for the noise gate.

Usage
-----
    uv run python noise_gate/enroll_target.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import sounddevice as sd
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from speechbrain.inference.speaker import EncoderClassifier

DURATION = 5          # seconds (as requested)
SAMPLE_RATE = 16_000  # record at 16 kHz directly (model rate)
CHANNELS = 1
TARGET_PATH = os.path.join(os.path.dirname(__file__), "target.npy")


def record_and_save(model: EncoderClassifier, device: torch.device, name: str, output_path: str):
    input(f"Press Enter, then have the {name} speak clearly for {DURATION} seconds … ")
    print(f"🎙  Recording {name} for {DURATION}s …")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    print(f"✓  Recording for {name} complete.\n")

    waveform = torch.from_numpy(audio[:, 0]).unsqueeze(0).float().to(device)

    with torch.no_grad():
        embedding = model.encode_batch(waveform)
        vec = embedding.squeeze().cpu().numpy()

    np.save(output_path, vec)
    print(f"✓  Saved {vec.shape[0]}-d embedding for {name} → {output_path}")
    print(f"   L2 norm: {np.linalg.norm(vec):.4f}\n")


def main() -> None:
    print("Loading ECAPA-TDNN model …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(
            os.path.dirname(__file__), "..", "pretrained_models", "spkrec-ecapa-voxceleb"
        ),
        run_opts={"device": str(device)},
    )
    print(f"✓  Model loaded on {device}\n")

    print("--- Enrolling Person X (Target) ---")
    record_and_save(model, device, "Target", TARGET_PATH)

    print(f"Enrollment complete. You can now run:  uv run python noise_gate/target_speaker_vad.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋  Cancelled.")
        sys.exit(0)
