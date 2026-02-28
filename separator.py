"""
Speech Separator using pre-trained SepFormer
=============================================

Wraps SpeechBrain's SepFormer (trained on WHAMR!) to blindly
separate a mixture into individual speaker streams.

No custom training required — uses pre-trained weights from
HuggingFace.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation

from config import SEPARATOR_SR, TARGET_SR

logger = logging.getLogger(__name__)


class SpeechSeparator:
    """Blind speech separator using pre-trained SepFormer.

    Parameters
    ----------
    source : str
        HuggingFace repo or local path for the pre-trained model.
        Default: ``speechbrain/sepformer-whamr`` (handles noise + reverb).
    save_dir : str | Path
        Local cache for downloaded weights.
    device : str | None
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        source: str = "speechbrain/sepformer-whamr",
        save_dir: str | Path = "pretrained_models/sepformer-whamr",
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        logger.info("Loading SepFormer from '%s' → %s", source, device)

        self.model = SepformerSeparation.from_hparams(
            source=source,
            savedir=str(save_dir),
            run_opts={"device": device},
        )
        self.model.eval()
        logger.info("SepFormer ready")

    @torch.no_grad()
    def separate(
        self,
        mixture: torch.Tensor,
        sr: int = TARGET_SR,
    ) -> list[torch.Tensor]:
        """Separate a mixture into individual speaker streams.

        Parameters
        ----------
        mixture : torch.Tensor
            1-D mono waveform at ``sr`` Hz.
        sr : int
            Sample rate of the input.

        Returns
        -------
        list[torch.Tensor]
            List of 1-D separated waveforms, resampled back to ``sr``.
            Typically 2 streams (SepFormer trained on 2-speaker mixtures).
        """
        wav = mixture.clone().float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        # SepFormer expects 8 kHz
        if sr != SEPARATOR_SR:
            wav = torchaudio.transforms.Resample(sr, SEPARATOR_SR)(wav)

        # Run separation
        est_sources = self.model.separate_batch(wav.to(self.device))
        # est_sources shape: [batch, time, n_sources]

        est_sources = est_sources.squeeze(0).cpu()  # [time, n_sources]

        # Split into individual streams and resample back
        streams = []
        for i in range(est_sources.shape[-1]):
            stream = est_sources[:, i]  # [time]
            if sr != SEPARATOR_SR:
                stream = torchaudio.transforms.Resample(SEPARATOR_SR, sr)(stream)
            streams.append(stream)

        logger.info("Separated mixture into %d streams", len(streams))
        return streams

    @torch.no_grad()
    def separate_chunk(
        self,
        chunk: torch.Tensor,
        sr: int = TARGET_SR,
    ) -> list[torch.Tensor]:
        """Separate a single chunk (alias for ``separate``).

        In a future version this could maintain state for streaming.
        For now it's a stateless call identical to ``separate``.
        """
        return self.separate(chunk, sr)
