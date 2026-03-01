"""
WeSep Target Speech Extraction (TSE) Experiment
==============================================

This script performs true reference-conditioned TSE:
  - Input A: mixed/overlapping speech audio
  - Input B: reference speech from one target speaker
  - Output : extracted target speaker audio

Back-end:
  - WeSep (downloaded locally in `third_party/wesep`)
  - Default model: WeSep english checkpoint (auto-downloaded on first run)

Examples
--------
Single target (one reference file):
    uv run python tse.py \
      --mix mix.wav \
      --ref ref.wav \
      --out out_tse.wav

Multiple targets (one output per reference):
    uv run python tse.py \
      --mix mix.wav \
      --refs-dir refs/ \
      --out-dir tse_outputs/

Rolling-window mode (realtime-style simulation):
    uv run python tse.py \
      --mix mix.wav \
      --ref ref.wav \
      --mode rolling \
      --window-sec 1.0 \
      --hop-sec 0.2 \
      --out out_tse_rolling.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]


SUPPORTED_REF_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
DEFAULT_MATCH_THRESHOLD = 0.35

_SPK_VERIFIER = None


@dataclass
class ExtractionStats:
    target_name: str
    mode: str
    audio_seconds: float
    wall_seconds: float
    rtf: float
    chunks: int
    mean_chunk_ms: float
    p95_chunk_ms: float
    max_chunk_ms: float
    match_score: float
    matched: bool


def resolve_device(device: str) -> str:
    """Resolve `auto` to a concrete torch device string."""
    if device != "auto":
        if device == "mps":
            # Current WeSep BSRNN checkpoint path is not stable on MPS due
            # explicit tensor-type usage in upstream model code.
            return "cpu"
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def add_wesep_to_pythonpath(wesep_dir: str) -> None:
    root = Path(wesep_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"WeSep directory not found: {root}")
    sys.path.insert(0, str(root))


def add_wespeaker_to_pythonpath(wespeaker_dir: str) -> None:
    root = Path(wespeaker_dir).resolve()
    if not root.exists():
        return
    sys.path.insert(0, str(root))


def import_wesep_loaders(wesep_dir: str, wespeaker_dir: str):
    add_wespeaker_to_pythonpath(wespeaker_dir)
    add_wesep_to_pythonpath(wesep_dir)
    try:
        from wesep import load_model, load_model_local  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import WeSep. Ensure dependencies are installed.\n"
            "Try:\n"
            "  uv pip install -e third_party/wesep\n"
            "If needed also install:\n"
            "  uv pip install silero-vad kaldiio soundfile pyyaml\n"
            f"Import error: {exc}"
        ) from exc
    return load_model, load_model_local


def load_audio_mono(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav, sr


def resample_audio(wav: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wav
    resampler = torchaudio.transforms.Resample(sr_in, sr_out)
    return resampler(wav)


def ensure_single_channel_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    """Force output tensor shape [1, length]."""
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav[:1]
    if wav.shape[1] < length:
        wav = F.pad(wav, (0, length - wav.shape[1]))
    elif wav.shape[1] > length:
        wav = wav[:, :length]
    return wav


def peak_normalize(wav: torch.Tensor, peak: float) -> torch.Tensor:
    mx = float(wav.abs().max().item()) if wav.numel() > 0 else 0.0
    if mx < 1e-9:
        return wav
    scale = min(1.0, peak / mx)
    return wav * scale


def save_audio(path: Path, wav: torch.Tensor, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.cpu(), sr)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((p / 100.0) * (len(vals) - 1)))
    return vals[max(0, min(len(vals) - 1, idx))]


def make_starts(total_samples: int, window_samples: int, hop_samples: int) -> list[int]:
    if total_samples <= window_samples:
        return [0]
    starts = list(range(0, total_samples - window_samples + 1, hop_samples))
    final_start = total_samples - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def get_speaker_verifier(device: str):
    """Lazy-load SpeechBrain ECAPA verifier."""
    global _SPK_VERIFIER
    if _SPK_VERIFIER is not None:
        return _SPK_VERIFIER

    from speechbrain.inference.speaker import EncoderClassifier

    # Keep verifier on stable devices; mps support is less consistent.
    verify_device = "cuda" if device == "cuda" else "cpu"
    _SPK_VERIFIER = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": verify_device},
    )
    return _SPK_VERIFIER


def compute_match_score(
    ref_wav: torch.Tensor,
    out_wav: torch.Tensor,
    sr: int,
    device: str,
) -> float:
    """Cosine similarity between reference and extracted speaker embeddings."""
    del sr
    model = get_speaker_verifier(device)
    verify_device = "cuda" if device == "cuda" else "cpu"

    ref = ref_wav.to(verify_device).float()
    out = out_wav.to(verify_device).float()

    with torch.no_grad():
        ref_emb = model.encode_batch(ref).squeeze()
        out_emb = model.encode_batch(out).squeeze()

    if ref_emb.ndim == 0 or out_emb.ndim == 0:
        return 0.0
    score = F.cosine_similarity(ref_emb.unsqueeze(0), out_emb.unsqueeze(0)).item()
    return float(score)


def load_extractor(
    wesep_dir: str,
    wespeaker_dir: str,
    model_dir: str,
    device: str,
    sample_rate: int,
    use_vad: bool,
):
    load_model, load_model_local = import_wesep_loaders(wesep_dir, wespeaker_dir)

    if model_dir:
        extractor = load_model_local(model_dir)
    else:
        # WeSep's english model auto-downloads to ~/.wesep/english on first run.
        extractor = load_model("english")

    extractor.set_device(device)
    extractor.set_resample_rate(sample_rate)
    extractor.set_vad(use_vad)
    extractor.set_wavform_norm(True)
    extractor.set_output_norm(False)
    return extractor


def extract_full(
    extractor,
    mix_wav: torch.Tensor,
    ref_wav: torch.Tensor,
    sr: int,
) -> tuple[torch.Tensor, list[float]]:
    t0 = time.perf_counter()
    out = extractor.extract_speech_from_pcm(mix_wav, sr, ref_wav, sr)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is None:
        out = torch.zeros_like(mix_wav)
    out = ensure_single_channel_length(out, mix_wav.shape[1])
    return out, [elapsed_ms]


def extract_rolling(
    extractor,
    mix_wav: torch.Tensor,
    ref_wav: torch.Tensor,
    sr: int,
    window_sec: float,
    hop_sec: float,
) -> tuple[torch.Tensor, list[float]]:
    total_samples = mix_wav.shape[1]
    window_samples = max(1, int(window_sec * sr))
    hop_samples = max(1, int(hop_sec * sr))

    starts = make_starts(total_samples, window_samples, hop_samples)
    out = torch.zeros(1, total_samples, dtype=torch.float32)
    norm = torch.zeros(1, total_samples, dtype=torch.float32)
    win = torch.hann_window(window_samples, dtype=torch.float32).unsqueeze(0)
    chunk_times: list[float] = []

    for start in starts:
        end = min(start + window_samples, total_samples)
        take = end - start
        chunk = mix_wav[:, start:end]
        if take < window_samples:
            chunk = F.pad(chunk, (0, window_samples - take))

        t0 = time.perf_counter()
        pred = extractor.extract_speech_from_pcm(chunk, sr, ref_wav, sr)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        chunk_times.append(elapsed_ms)

        if pred is None:
            pred = torch.zeros_like(chunk)
        pred = ensure_single_channel_length(pred, window_samples)

        out[:, start:end] += pred[:, :take] * win[:, :take]
        norm[:, start:end] += win[:, :take] * win[:, :take]

    out = out / torch.clamp(norm, min=1e-8)
    return out, chunk_times


def build_stats(
    target_name: str,
    mode: str,
    audio_seconds: float,
    wall_seconds: float,
    chunk_times_ms: list[float],
    match_score: float,
    matched: bool,
) -> ExtractionStats:
    if chunk_times_ms:
        mean_ms = sum(chunk_times_ms) / len(chunk_times_ms)
        p95_ms = percentile(chunk_times_ms, 95.0)
        max_ms = max(chunk_times_ms)
    else:
        mean_ms = 0.0
        p95_ms = 0.0
        max_ms = 0.0

    rtf = wall_seconds / audio_seconds if audio_seconds > 0 else 0.0
    return ExtractionStats(
        target_name=target_name,
        mode=mode,
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        rtf=rtf,
        chunks=len(chunk_times_ms),
        mean_chunk_ms=mean_ms,
        p95_chunk_ms=p95_ms,
        max_chunk_ms=max_ms,
        match_score=match_score,
        matched=matched,
    )


def iter_reference_files(refs_dir: str) -> list[Path]:
    root = Path(refs_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {refs_dir}")
    files = sorted(
        f for f in root.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_REF_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No reference audio files found in {refs_dir}. "
            f"Supported: {', '.join(sorted(SUPPORTED_REF_EXTENSIONS))}"
        )
    return files


def run_one_target(
    extractor,
    mix_wav: torch.Tensor,
    mix_sr: int,
    ref_path: Path,
    mode: str,
    window_sec: float,
    hop_sec: float,
    peak: float,
    match_threshold: float,
    device: str,
) -> tuple[torch.Tensor, ExtractionStats]:
    ref_wav, ref_sr = load_audio_mono(str(ref_path))
    ref_wav = resample_audio(ref_wav, ref_sr, mix_sr)

    t0 = time.perf_counter()
    if mode == "full":
        out_wav, chunk_times = extract_full(extractor, mix_wav, ref_wav, mix_sr)
    else:
        out_wav, chunk_times = extract_rolling(
            extractor=extractor,
            mix_wav=mix_wav,
            ref_wav=ref_wav,
            sr=mix_sr,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )
    wall = time.perf_counter() - t0

    out_wav = peak_normalize(out_wav, peak=peak)
    match_score = compute_match_score(ref_wav, out_wav, mix_sr, device=device)
    matched = match_score >= match_threshold
    stats = build_stats(
        target_name=ref_path.stem,
        mode=mode,
        audio_seconds=mix_wav.shape[1] / mix_sr,
        wall_seconds=wall,
        chunk_times_ms=chunk_times,
        match_score=match_score,
        matched=matched,
    )
    return out_wav, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reference-conditioned TSE experiment using WeSep."
    )
    parser.add_argument("--mix", required=True, help="Path to mixed/overlapped audio.")

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--ref", help="Single reference audio path.")
    target_group.add_argument(
        "--refs-dir",
        help="Directory of reference audios (one output per file).",
    )

    parser.add_argument(
        "--mode",
        choices=["full", "rolling"],
        default="rolling",
        help="`full` runs one-shot extraction, `rolling` simulates realtime chunked inference.",
    )
    parser.add_argument("--window-sec", type=float, default=1.0, help="Rolling window size.")
    parser.add_argument("--hop-sec", type=float, default=0.2, help="Rolling hop size.")

    parser.add_argument("--sample-rate", type=int, default=16000, help="Model sample rate.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--vad", action="store_true", help="Enable VAD on enrollment reference.")
    parser.add_argument("--peak", type=float, default=0.95, help="Peak normalize output to this value.")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help=f"Speaker match threshold for printing 'no match' (default: {DEFAULT_MATCH_THRESHOLD}).",
    )

    parser.add_argument(
        "--wesep-dir",
        default="third_party/wesep",
        help="Local WeSep repository path.",
    )
    parser.add_argument(
        "--wespeaker-dir",
        default="third_party/wespeaker",
        help="Local WeSpeaker repository path used by WeSep speaker encoders.",
    )
    parser.add_argument(
        "--model-dir",
        default="",
        help="Optional local WeSep checkpoint dir with config.yaml + avg_model.pt. "
        "If omitted, uses WeSep english auto-download.",
    )

    parser.add_argument("--out", default="out_tse.wav", help="Output path for single-target mode.")
    parser.add_argument("--out-dir", default="tse_outputs", help="Output dir for multi-target mode.")
    parser.add_argument("--stats-path", default="", help="Optional JSON stats path (single-target).")
    parser.add_argument(
        "--stats-dir",
        default="",
        help="Optional stats dir (multi-target). Defaults to <out-dir>/stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "rolling" and args.hop_sec > args.window_sec:
        raise ValueError("--hop-sec must be <= --window-sec in rolling mode.")

    device = resolve_device(args.device)
    if args.device == "mps":
        print("Note: WeSep BSRNN is not stable on MPS; falling back to CPU.")
    elif args.device == "auto" and torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("Note: MPS detected, but using CPU for WeSep compatibility.")
    print("=" * 68)
    print("WeSep Target Speech Extraction")
    print("=" * 68)
    print(f"Mixture:   {args.mix}")
    print(f"Device:    {device}")
    print(f"Mode:      {args.mode}")
    print(f"WeSep dir: {Path(args.wesep_dir).resolve()}")
    print(f"WeSpeaker: {Path(args.wespeaker_dir).resolve()}")
    if args.model_dir:
        print(f"Model dir: {Path(args.model_dir).resolve()}")
    else:
        print("Model dir: auto-download (english bsrnn_ecapa_vox1)")

    extractor = load_extractor(
        wesep_dir=args.wesep_dir,
        wespeaker_dir=args.wespeaker_dir,
        model_dir=args.model_dir,
        device=device,
        sample_rate=args.sample_rate,
        use_vad=args.vad,
    )

    mix_wav, mix_sr = load_audio_mono(args.mix)
    mix_wav = resample_audio(mix_wav, mix_sr, args.sample_rate)
    mix_sr = args.sample_rate
    print(f"Loaded mixture: {mix_wav.shape[1] / mix_sr:.2f}s @ {mix_sr}Hz")

    if args.ref:
        out_wav, stats = run_one_target(
            extractor=extractor,
            mix_wav=mix_wav,
            mix_sr=mix_sr,
            ref_path=Path(args.ref),
            mode=args.mode,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            peak=args.peak,
            match_threshold=args.match_threshold,
            device=device,
        )
        out_path = Path(args.out)
        save_audio(out_path, out_wav, mix_sr)
        print(f"Saved output: {out_path}")
        if stats.matched:
            print(f"match score={stats.match_score:.3f} (threshold={args.match_threshold:.3f})")
        else:
            print("no match")
            print(f"match score={stats.match_score:.3f} (threshold={args.match_threshold:.3f})")
        print(
            f"RTF={stats.rtf:.3f} | chunks={stats.chunks} | "
            f"mean={stats.mean_chunk_ms:.1f}ms | p95={stats.p95_chunk_ms:.1f}ms"
        )

        if args.stats_path:
            stats_path = Path(args.stats_path)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
            print(f"Saved stats: {stats_path}")
        return

    ref_files = iter_reference_files(args.refs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_dir = Path(args.stats_dir) if args.stats_dir else out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[ExtractionStats] = []
    for ref_file in ref_files:
        print(f"\n[Target] {ref_file.name}")
        out_wav, stats = run_one_target(
            extractor=extractor,
            mix_wav=mix_wav,
            mix_sr=mix_sr,
            ref_path=ref_file,
            mode=args.mode,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            peak=args.peak,
            match_threshold=args.match_threshold,
            device=device,
        )
        out_path = out_dir / f"{ref_file.stem}_tse.wav"
        save_audio(out_path, out_wav, mix_sr)
        match_tag = "match" if stats.matched else "no match"
        print(
            f"  {match_tag} | score={stats.match_score:.3f} | saved={out_path} | rtf={stats.rtf:.3f} | "
            f"mean={stats.mean_chunk_ms:.1f}ms | p95={stats.p95_chunk_ms:.1f}ms"
        )

        per_target_stats = stats_dir / f"{ref_file.stem}.json"
        per_target_stats.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
        all_stats.append(stats)

    summary = {
        "mix": str(Path(args.mix).resolve()),
        "targets": len(all_stats),
        "mode": args.mode,
        "sample_rate": mix_sr,
        "mean_rtf": sum(s.rtf for s in all_stats) / len(all_stats) if all_stats else 0.0,
        "mean_chunk_ms": (
            sum(s.mean_chunk_ms for s in all_stats) / len(all_stats) if all_stats else 0.0
        ),
        "matches": sum(1 for s in all_stats if s.matched),
        "match_threshold": args.match_threshold,
    }
    summary_path = stats_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
