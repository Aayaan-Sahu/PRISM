"""
Voice Activity Detection wrapper using Silero VAD.

Provides utilities to detect speech regions and strip silence from
audio waveforms, ensuring enrollment embeddings are computed from
clean speech only.
"""

from __future__ import annotations

import logging
from typing import Union

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Silero VAD operates at 16 kHz
_SILERO_SR = 16_000


class VoiceActivityDetector:
    """Lightweight wrapper around the Silero VAD model.

    Parameters
    ----------
    threshold : float
        Speech probability threshold (0–1). Frames above this are
        considered speech. Default 0.5 works well for most cases.
    min_speech_ms : int
        Minimum speech segment length in milliseconds. Shorter
        segments are discarded.
    min_silence_ms : int
        Minimum silence length between speech segments before they
        are split.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
    ) -> None:
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

        logger.info("Loading Silero VAD …")
        self.model, self.utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self._get_speech_timestamps = self.utils[0]
        logger.info("Silero VAD ready")

    def get_speech_timestamps(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> list[dict[str, int]]:
        """Detect speech regions in a waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            1-D or 2-D (1, T) float tensor.
        sr : int
            Sample rate of the waveform.

        Returns
        -------
        list[dict]
            Each dict has ``"start"`` and ``"end"`` keys (sample indices
            at the *original* sample rate).
        """
        wav = waveform.clone().float()
        if wav.ndim == 2:
            if wav.shape[0] > 1:
                # Multi-channel → mono mix
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)

        # Resample to 16 kHz if needed
        if sr != _SILERO_SR:
            wav = torchaudio.transforms.Resample(sr, _SILERO_SR)(wav)

        timestamps = self._get_speech_timestamps(
            wav,
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            sampling_rate=_SILERO_SR,
        )

        # Convert sample indices back to original sample rate
        if sr != _SILERO_SR:
            ratio = sr / _SILERO_SR
            timestamps = [
                {"start": int(t["start"] * ratio), "end": int(t["end"] * ratio)}
                for t in timestamps
            ]

        return timestamps

    def trim_silence(
        self,
        waveform: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Remove silence from a waveform, keeping only speech.

        Parameters
        ----------
        waveform : torch.Tensor
            1-D or 2-D (1, T) float tensor.
        sr : int
            Sample rate.

        Returns
        -------
        torch.Tensor
            1-D tensor with only speech regions concatenated.
        """
        wav = waveform.clone().float()
        if wav.ndim == 2:
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)

        timestamps = self.get_speech_timestamps(wav, sr)

        if not timestamps:
            logger.warning("No speech detected — returning original waveform")
            return wav

        speech_chunks = [wav[t["start"] : t["end"]] for t in timestamps]
        trimmed = torch.cat(speech_chunks)

        total_sec = len(wav) / sr
        speech_sec = len(trimmed) / sr
        logger.info(
            "VAD: %.1fs total → %.1fs speech (%.0f%% kept)",
            total_sec,
            speech_sec,
            100 * speech_sec / total_sec,
        )

        return trimmed
