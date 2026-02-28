"""
Speaker Embedding Extraction via ECAPA-TDNN
============================================

Extracts speaker-specific d-vectors (embeddings) that encode the unique
physiological characteristics of a speaker's voice — invariant to phonetic
content and environmental noise.

Architecture : ECAPA-TDNN  (speechbrain.lobes.models.ECAPA_TDNN)
Weights      : speechbrain/spkrec-ecapa-voxceleb  (HuggingFace)
Embedding dim: 192

Usage
-----
    encoder = SpeakerEncoder()                      # downloads weights on first run
    emb_a   = encoder.encode("speaker_a.wav")       # → torch.Tensor [192]
    emb_b   = encoder.encode("speaker_b.wav")
    score   = cosine_similarity(emb_a, emb_b)       # 1.0 = identical, 0.0 = unrelated
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

# ─── Logging ─────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# ─── Constants ───────────────────────────────────────────────────
TARGET_SAMPLE_RATE: int = 16_000          # Hz — required by the pre-trained model
EMBEDDING_DIM: int = 192                  # ECAPA-TDNN output dimensionality
SUPPORTED_EXTENSIONS: set[str] = {".wav", ".flac", ".ogg", ".mp3"}

# Default verification threshold (cosine similarity).
# Scores above this value are considered **same speaker**.
# Tuned on VoxCeleb1-O cleaned trial list; adjust per deployment.
VERIFICATION_THRESHOLD: float = 0.25


# ─── SpeakerEncoder ─────────────────────────────────────────────
class SpeakerEncoder:
    """High-level wrapper around the ECAPA-TDNN speaker encoder.

    Parameters
    ----------
    source : str
        HuggingFace repo ID or local directory containing the pre-trained
        model.  Defaults to ``"speechbrain/spkrec-ecapa-voxceleb"``.
    save_dir : str | Path
        Local cache directory for downloaded weights.
    device : str
        ``"cpu"``, ``"cuda"``, or ``"mps"``.  Auto-detected when omitted.
    """

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        save_dir: Union[str, Path] = "pretrained_models/spkrec-ecapa-voxceleb",
        device: str | None = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        logger.info("Loading ECAPA-TDNN from '%s' → %s", source, self.device)

        self.model = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(save_dir),
            run_opts={"device": str(self.device)},
        )
        # Ensure evaluation mode and frozen weights.
        self.model.eval()
        logger.info("Model ready  (embedding dim = %d)", EMBEDDING_DIM)

    # ── Preprocessing ────────────────────────────────────────────
    def _load_audio(self, audio: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        """Load and normalise audio to 16 kHz mono float32.

        Accepts a file path **or** a raw waveform tensor.

        Returns
        -------
        torch.Tensor
            Shape ``[1, num_samples]`` — batch-ready.
        """
        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported audio format '{path.suffix}'. "
                    f"Expected one of {SUPPORTED_EXTENSIONS}."
                )

            try:
                waveform, sr = torchaudio.load(str(path))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to decode audio file '{path}': {exc}"
                ) from exc
        elif isinstance(audio, torch.Tensor):
            waveform = audio.clone()
            sr = TARGET_SAMPLE_RATE  # assume correct SR for raw tensors
        else:
            raise TypeError(
                f"Expected file path or torch.Tensor, got {type(audio).__name__}"
            )

        # Mono-mix if multi-channel.
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz when necessary.
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Peak-normalise to [-1, 1] (prevents volume bias).
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        return waveform  # [1, T]

    # ── Embedding extraction ─────────────────────────────────────
    @torch.no_grad()
    def encode(
        self,
        audio: Union[str, Path, torch.Tensor],
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Extract a fixed-length speaker embedding (d-vector).

        Parameters
        ----------
        audio : str | Path | torch.Tensor
            File path (.wav, .flac, .ogg, .mp3) **or** a waveform tensor.
        return_numpy : bool
            If ``True``, return a 1-D NumPy array instead of a Torch tensor.

        Returns
        -------
        torch.Tensor | np.ndarray
            Speaker embedding of shape ``[192]``.
        """
        waveform = self._load_audio(audio)                    # [1, T]
        waveform = waveform.to(self.device)

        # SpeechBrain's EncoderClassifier expects [batch, time].
        embedding = self.model.encode_batch(waveform)         # [1, 1, 192]
        embedding = embedding.squeeze()                       # [192]

        if return_numpy:
            return embedding.cpu().numpy()
        return embedding.cpu()

    # ── Batch encoding ───────────────────────────────────────────
    @torch.no_grad()
    def encode_batch(
        self,
        audio_list: list[Union[str, Path, torch.Tensor]],
        return_numpy: bool = False,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Encode multiple utterances and return stacked embeddings.

        Returns
        -------
        torch.Tensor | np.ndarray
            Shape ``[N, 192]``.
        """
        embeddings = [self.encode(a) for a in audio_list]
        stacked = torch.stack(embeddings)                     # [N, 192]
        if return_numpy:
            return stacked.numpy()
        return stacked


# ─── Verification Utilities ──────────────────────────────────────
def cosine_similarity(
    emb_a: Union[torch.Tensor, np.ndarray],
    emb_b: Union[torch.Tensor, np.ndarray],
) -> float:
    """Compute cosine similarity between two speaker embeddings.

    Returns
    -------
    float
        Value in ``[-1, 1]``.  Higher → more likely same speaker.
    """
    if isinstance(emb_a, np.ndarray):
        emb_a = torch.from_numpy(emb_a)
    if isinstance(emb_b, np.ndarray):
        emb_b = torch.from_numpy(emb_b)

    emb_a = emb_a.flatten().float()
    emb_b = emb_b.flatten().float()

    return torch.nn.functional.cosine_similarity(
        emb_a.unsqueeze(0), emb_b.unsqueeze(0)
    ).item()


def verify_speaker(
    emb_a: Union[torch.Tensor, np.ndarray],
    emb_b: Union[torch.Tensor, np.ndarray],
    threshold: float = VERIFICATION_THRESHOLD,
) -> tuple[bool, float]:
    """Determine whether two embeddings belong to the same speaker.

    Parameters
    ----------
    emb_a, emb_b : Tensor | ndarray
        Speaker embeddings of shape ``[192]``.
    threshold : float
        Decision boundary.  Pairs scoring above this value are
        accepted as the **same speaker**.

    Returns
    -------
    same_speaker : bool
    score : float
        Raw cosine-similarity score.
    """
    score = cosine_similarity(emb_a, emb_b)
    return score >= threshold, score


def main() -> None:
    """Quick demonstration: record two speakers and compare them."""
    import sys
    import sounddevice as sd

    print(
        "Speaker Embedding Extraction\n"
        "Compares two live recorded voices and reports whether they are\n"
        "from the same speaker.\n"
    )

    # ── Initialise encoder ───────────────────────────────────────
    encoder = SpeakerEncoder()

    duration = 5.0
    fs = 16000

    def record_speaker(name: str) -> torch.Tensor:
        input(f"\n🎙  Press Enter to start recording {name} for {duration} seconds...")
        print(f"🔴 Recording {name}...")
        try:
            # Using system default microphone (mono, 16kHz)
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            print("✅ Recording complete.")
            return torch.from_numpy(audio).T  # [1, T] sequence
        except Exception as e:
            print(f"Failed to record audio: {e}")
            sys.exit(1)

    # ── Record and Extract embeddings for Speaker A ────────────────
    audio_a = record_speaker("Speaker A")
    print(f"🎙  Encoding Speaker A …")
    emb_a = encoder.encode(audio_a)
    print(f"    → embedding shape: {tuple(emb_a.shape)}")

    # ── Record and Extract embeddings for Speaker B ────────────────
    audio_b = record_speaker("Speaker B")
    print(f"🎙  Encoding Speaker B …")
    emb_b = encoder.encode(audio_b)
    print(f"    → embedding shape: {tuple(emb_b.shape)}")

    # ── Compare ──────────────────────────────────────────────────
    same, score = verify_speaker(emb_a, emb_b)

    print("\n" + "═" * 52)
    print(f"  Cosine similarity : {score:+.4f}")
    print(f"  Threshold         : {VERIFICATION_THRESHOLD:+.4f}")
    print(f"  Verdict           : {'✅ SAME speaker' if same else '❌ DIFFERENT speakers'}")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    main()
