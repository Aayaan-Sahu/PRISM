"""
Speaker Identifier
==================

Matches separated audio streams to enrolled speaker voiceprints
using ECAPA-TDNN embeddings and cosine similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from config import ID_THRESHOLD, VOICEPRINT_DIR
from embeddings import SpeakerEncoder, cosine_similarity

logger = logging.getLogger(__name__)


class SpeakerIdentifier:
    """Identifies which separated streams belong to enrolled speakers.

    Parameters
    ----------
    voiceprint_dir : Path
        Directory containing enrolled voiceprints (``*.pt`` files).
    threshold : float
        Cosine similarity threshold. Streams scoring above this
        against an enrolled voiceprint are tagged as that speaker.
    encoder : SpeakerEncoder | None
        Shared encoder instance. Created if ``None``.
    """

    def __init__(
        self,
        voiceprint_dir: Path = VOICEPRINT_DIR,
        threshold: float = ID_THRESHOLD,
        encoder: SpeakerEncoder | None = None,
    ) -> None:
        self.threshold = threshold
        self.encoder = encoder or SpeakerEncoder()
        self.voiceprints: dict[str, torch.Tensor] = {}

        self._load_voiceprints(voiceprint_dir)

    def _load_voiceprints(self, directory: Path) -> None:
        """Load all .pt voiceprints from the given directory."""
        directory = Path(directory)
        if not directory.exists():
            logger.warning("Voiceprint directory '%s' not found", directory)
            return

        for pt_file in sorted(directory.glob("*.pt")):
            name = pt_file.stem
            emb = torch.load(pt_file, map_location="cpu", weights_only=True)
            self.voiceprints[name] = emb.float()
            logger.info("Loaded voiceprint: %s (%s)", name, tuple(emb.shape))

        if not self.voiceprints:
            logger.warning("No voiceprints found in '%s'", directory)
        else:
            logger.info(
                "Loaded %d voiceprint(s): %s",
                len(self.voiceprints),
                ", ".join(self.voiceprints.keys()),
            )

    @property
    def enrolled_speakers(self) -> list[str]:
        """Names of all enrolled speakers."""
        return list(self.voiceprints.keys())

    def identify_stream(
        self,
        stream_waveform: torch.Tensor,
    ) -> tuple[str | None, float]:
        """Identify the speaker in a single separated audio stream.

        Parameters
        ----------
        stream_waveform : torch.Tensor
            1-D waveform of a single separated speaker at 16 kHz.

        Returns
        -------
        tuple[str | None, float]
            ``(speaker_name, score)`` if matched above threshold,
            or ``(None, best_score)`` if no match.
        """
        if not self.voiceprints:
            return None, 0.0

        # Compute embedding for this stream
        stream_emb = self.encoder.encode(stream_waveform)

        best_name: str | None = None
        best_score: float = -1.0

        for name, vp in self.voiceprints.items():
            score = cosine_similarity(stream_emb, vp)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= self.threshold:
            logger.debug("Stream → %s (score=%.3f)", best_name, best_score)
            return best_name, best_score
        else:
            logger.debug(
                "Stream → unknown (best=%s, score=%.3f < threshold=%.3f)",
                best_name,
                best_score,
                self.threshold,
            )
            return None, best_score

    def identify_streams(
        self,
        streams: list[torch.Tensor],
    ) -> list[tuple[str | None, float]]:
        """Identify speakers across multiple separated streams.

        Parameters
        ----------
        streams : list[torch.Tensor]
            List of 1-D separated waveforms.

        Returns
        -------
        list[tuple[str | None, float]]
            One ``(speaker_name | None, score)`` per stream.
        """
        results = []
        for i, stream in enumerate(streams):
            name, score = self.identify_stream(stream)
            tag = name or "unknown"
            logger.info("  Stream %d → %s (score=%.3f)", i, tag, score)
            results.append((name, score))
        return results
