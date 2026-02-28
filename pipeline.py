"""
Live Selective Speaker Amplification Pipeline
===============================================

Streams audio from the microphone, separates speakers using SepFormer,
identifies enrolled speakers via ECAPA-TDNN embeddings, and selectively
amplifies them while suppressing unknown speakers.

Usage
-----
    python pipeline.py

Prerequisites: enroll at least one speaker first:
    python enroll.py --name aayaan --angle -45
"""

from __future__ import annotations

import logging
import signal
import sys
import time

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from config import (
    AGG_DEVICE_INDEX,
    CHANNELS,
    CHUNK_SEC,
    G_BG,
    G_FG,
    ID_THRESHOLD,
    OVERLAP_SEC,
    SMOOTHING_TAU,
    TARGET_SR,
    VOICEPRINT_DIR,
)
from embeddings import SpeakerEncoder
from identifier import SpeakerIdentifier
from separator import SpeechSeparator
from vad import VoiceActivityDetector

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)


class LivePipeline:
    """Real-time selective speaker amplification.

    Parameters
    ----------
    device_index : int
        Audio input device index.
    channels : int
        Number of input channels.
    """

    def __init__(
        self,
        device_index: int = AGG_DEVICE_INDEX,
        channels: int = CHANNELS,
    ) -> None:
        self.device_index = device_index
        self.channels = channels

        # Query device for native sample rate
        info = sd.query_devices(device_index, "input")
        self.native_sr = int(info["default_samplerate"])

        # Chunk sizes
        self.chunk_samples_native = int(self.native_sr * CHUNK_SEC)
        self.chunk_samples_16k = int(TARGET_SR * CHUNK_SEC)

        # Resampler: native → 16 kHz
        self.resample_down = torchaudio.transforms.Resample(
            self.native_sr, TARGET_SR
        )
        # Resampler: 16 kHz → native (for output)
        self.resample_up = torchaudio.transforms.Resample(
            TARGET_SR, self.native_sr
        )

        # Accumulation buffer (native sample rate)
        self.buffer = np.zeros(0, dtype=np.float32)

        # Output buffer for playback
        self.output_chunks: list[np.ndarray] = []

        # Smooth gain per stream (initialised to foreground gain)
        self._smooth_gains: list[float] = []

        # ── Load models ──────────────────────────────────────────
        print("\n🔧 Loading models …\n")

        self.encoder = SpeakerEncoder()
        self.separator = SpeechSeparator()
        self.vad = VoiceActivityDetector()
        self.identifier = SpeakerIdentifier(
            voiceprint_dir=VOICEPRINT_DIR,
            threshold=ID_THRESHOLD,
            encoder=self.encoder,
        )

        if not self.identifier.enrolled_speakers:
            print("⚠️  No enrolled speakers found in voiceprints/")
            print("   Run:  python enroll.py --name <name> --angle <deg>")
            sys.exit(1)

        print(f"\n🎯 Enrolled speakers: {', '.join(self.identifier.enrolled_speakers)}")
        print(f"   Foreground gain: {G_FG:.2f}  |  Background gain: {G_BG:.2f}")
        print(f"   ID threshold: {ID_THRESHOLD:.2f}")
        print(f"   Chunk: {CHUNK_SEC}s  |  Device: {self.device_index} @ {self.native_sr} Hz\n")

    def _downmix_to_mono(self, multichannel: np.ndarray) -> np.ndarray:
        """Downmix 4-channel input to mono."""
        if multichannel.ndim == 1:
            return multichannel
        # Average all channels for a simple mono mix
        return multichannel.mean(axis=1).astype(np.float32)

    def _smooth_gain(self, target: float, idx: int) -> float:
        """Exponentially smooth the gain for stream *idx*."""
        while len(self._smooth_gains) <= idx:
            self._smooth_gains.append(G_FG)

        alpha = 1.0 - np.exp(-CHUNK_SEC / max(SMOOTHING_TAU, 1e-6))
        self._smooth_gains[idx] += alpha * (target - self._smooth_gains[idx])
        return self._smooth_gains[idx]

    def process_chunk(self, chunk_mono: np.ndarray) -> np.ndarray:
        """Process one audio chunk through the full pipeline.

        Parameters
        ----------
        chunk_mono : np.ndarray
            Mono float32 array at native sample rate.

        Returns
        -------
        np.ndarray
            Processed mono float32 array at native sample rate.
        """
        # Resample to 16 kHz for models
        wav_16k = self.resample_down(
            torch.from_numpy(chunk_mono).float()
        )

        # ── VAD: check if there's speech ─────────────────────────
        timestamps = self.vad.get_speech_timestamps(wav_16k, TARGET_SR)
        if not timestamps:
            # No speech — pass through at background gain
            logger.debug("No speech detected — passing through attenuated")
            return chunk_mono * G_BG

        # ── Separate speakers ────────────────────────────────────
        try:
            streams = self.separator.separate(wav_16k, sr=TARGET_SR)
        except Exception as e:
            logger.warning("Separation failed: %s — passing through raw", e)
            return chunk_mono  # fail-safe: raw audio

        if not streams:
            return chunk_mono * G_BG

        # ── Identify and apply gains ─────────────────────────────
        identities = self.identifier.identify_streams(streams)

        mixed = torch.zeros_like(wav_16k)
        enrolled_found = False

        for i, (stream, (name, score)) in enumerate(zip(streams, identities)):
            # Determine target gain
            if name is not None:
                target_gain = G_FG
                enrolled_found = True
            else:
                target_gain = G_BG

            gain = self._smooth_gain(target_gain, i)

            # Ensure stream matches the expected length
            if len(stream) > len(mixed):
                stream = stream[: len(mixed)]
            elif len(stream) < len(mixed):
                padded = torch.zeros_like(mixed)
                padded[: len(stream)] = stream
                stream = padded

            mixed += gain * stream

        # ── Peak limiter ─────────────────────────────────────────
        peak = mixed.abs().max()
        if peak > 1.0:
            mixed = mixed / peak

        # Resample back to native rate
        output = self.resample_up(mixed).numpy()

        # Match original chunk length
        if len(output) > len(chunk_mono):
            output = output[: len(chunk_mono)]
        elif len(output) < len(chunk_mono):
            padded = np.zeros(len(chunk_mono), dtype=np.float32)
            padded[: len(output)] = output
            output = padded

        # Log what happened
        tags = [name or "unknown" for name, _ in identities]
        scores = [f"{score:.2f}" for _, score in identities]
        logger.info(
            "Chunk processed | speakers: %s | scores: %s | enrolled: %s",
            tags, scores, enrolled_found,
        )

        return output.astype(np.float32)

    def run(self, duration_sec: float | None = None) -> None:
        """Start the live pipeline.

        Parameters
        ----------
        duration_sec : float | None
            If given, run for this many seconds then stop.
            If ``None``, run until Ctrl+C.
        """
        running = True

        def on_sigint(sig, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, on_sigint)

        print("🎧 Pipeline running — speak into the microphone")
        if duration_sec:
            print(f"   Auto-stopping after {duration_sec}s")
        print("   Press Ctrl+C to stop\n")

        start_time = time.time()
        chunk_native = int(self.native_sr * CHUNK_SEC)

        def callback(indata, frames, time_info, status):
            nonlocal running
            if not running:
                return

            # Accumulate samples
            mono = self._downmix_to_mono(
                np.asarray(indata[:, : self.channels], dtype=np.float32)
            )
            self.buffer = np.concatenate([self.buffer, mono])

            # Process when we have a full chunk
            while len(self.buffer) >= chunk_native:
                chunk = self.buffer[:chunk_native]
                self.buffer = self.buffer[chunk_native:]

                processed = self.process_chunk(chunk)
                self.output_chunks.append(processed)

        try:
            with sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.native_sr,
                blocksize=1024,
                dtype="float32",
                callback=callback,
            ):
                while running:
                    if duration_sec and (time.time() - start_time) >= duration_sec:
                        break
                    sd.sleep(100)
        except KeyboardInterrupt:
            pass

        print("\n\n🛑 Pipeline stopped")

        # Save output
        if self.output_chunks:
            import wave

            output = np.concatenate(self.output_chunks)
            filename = "pipeline_output.wav"

            with wave.open(filename, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.native_sr)
                pcm = np.clip(output * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(pcm.tobytes())

            secs = len(output) / self.native_sr
            print(f"💾 Saved {secs:.1f}s of processed audio → {filename}")
        else:
            print("   No audio was processed.")


# ─── CLI ─────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live selective speaker amplification."
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Duration in seconds (default: run until Ctrl+C)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=AGG_DEVICE_INDEX,
        help=f"Audio device index (default: {AGG_DEVICE_INDEX})",
    )
    parser.add_argument(
        "--fg-gain",
        type=float,
        default=G_FG,
        help=f"Foreground (enrolled) gain (default: {G_FG})",
    )
    parser.add_argument(
        "--bg-gain",
        type=float,
        default=G_BG,
        help=f"Background (unknown) gain (default: {G_BG})",
    )

    args = parser.parse_args()

    # Allow CLI overrides
    import config
    config.G_FG = args.fg_gain
    config.G_BG = args.bg_gain

    pipeline = LivePipeline(device_index=args.device)
    pipeline.run(duration_sec=args.duration)


if __name__ == "__main__":
    main()
