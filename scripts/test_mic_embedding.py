"""
Mic → Speaker Embedding  (quick CLI test)
==========================================

Press Enter, speak for 4 seconds, and see your 192-dim embedding.

Usage
-----
    pip install sounddevice speechbrain torch torchaudio
    python test_mic_embedding.py
"""

from __future__ import annotations

import io
import sys
import wave
import logging

import numpy as np
import sounddevice as sd

# ── Config ───────────────────────────────────────────────────────────
DURATION   = 4        # seconds of recording
SAMPLE_RATE = 16_000  # 16 kHz mono — what the model expects
CHANNELS   = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("mic-test")


def record_wav_bytes(duration: float = DURATION) -> bytes:
    """Record from the default mic and return raw .wav bytes."""
    print(f"\n🎙  Recording for {duration}s … speak now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()  # block until recording finishes
    print("✓  Recording complete.\n")

    # Pack the numpy array into a proper WAV byte buffer
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def wav_bytes_to_tensor(wav_bytes: bytes) -> "torch.Tensor":
    """Decode WAV bytes into a (1, num_samples) float32 tensor using stdlib only."""
    import torch

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    # 16-bit PCM → float32 in [-1, 1]
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples).unsqueeze(0)  # (1, T)


def main() -> None:
    # Lazy-import the heavy stuff so the prompt appears fast
    print("Loading ECAPA-TDNN model (first run downloads ~90 MB) …")

    import torch
    import torchaudio

    # ── Compatibility shim ───────────────────────────────────────
    # SpeechBrain 1.0.x calls torchaudio.list_audio_backends() on
    # import, but that function was removed in torchaudio ≥ 2.5.
    # Patch it back in so the import succeeds.
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg"]

    from speechbrain.inference.speaker import EncoderClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="./pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)},
    )
    print(f"✓  Model loaded on {device}\n")

    while True:
        input("Press Enter to record (Ctrl-C to quit) … ")

        wav_bytes = record_wav_bytes()

        # Decode WAV using stdlib (avoids torchaudio.load / torchcodec / FFmpeg)
        waveform = wav_bytes_to_tensor(wav_bytes).to(device)

        with torch.no_grad():
            embedding = model.encode_batch(waveform)
            vec = embedding.squeeze().cpu().tolist()

        # Show results
        print(f"Embedding dimensions : {len(vec)}")
        print(f"First 10 values      : {vec[:10]}")
        print(f"Min / Max            : {min(vec):.4f} / {max(vec):.4f}")
        print(f"Norm (L2)            : {sum(v**2 for v in vec) ** 0.5:.4f}")
        print("─" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋  Bye!")
        sys.exit(0)
