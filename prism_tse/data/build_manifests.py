"""Build JSONL manifests from downloaded corpora.

    python -m prism_tse.data.build_manifests --data_root data_root

Outputs under {data_root}/manifests:
  train_speech.jsonl  — LibriSpeech train-clean-100 (+360, + VCTK if present)
  val_speech.jsonl    — LibriSpeech dev-clean (speaker-disjoint from train)
  noise_train.jsonl / noise_val.jsonl — MUSAN noise+music (+ DNS if present),
                        file-disjoint 95/5 split
Rows: {"path", "speaker_id", "duration_s", "sr"}  (noise rows: no speaker_id).

Non-16 kHz sources (DNS 48k, VCTK 48k) are converted once into
{data_root}/prepared/ and manifests point at the converted files.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import soundfile as sf

from prism_tse.datasets.manifest import save_jsonl

SR = 16_000


def _info(path: Path):
    i = sf.info(str(path))
    return i.samplerate, i.frames / i.samplerate


def _convert_to_16k(src: Path, dst: Path) -> None:
    import torch
    import torchaudio

    dst.parent.mkdir(parents=True, exist_ok=True)
    wav, sr = torchaudio.load(str(src))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    sf.write(str(dst), wav.squeeze(0).clamp(-1, 1).numpy(), SR)


def _entry(path: Path, data_root: Path, prepared: Path, speaker_id: str | None):
    sr, dur = _info(path)
    out_path = path
    if sr != SR:
        rel = path.relative_to(data_root / "raw")
        out_path = (prepared / rel).with_suffix(".wav")
        if not out_path.exists():
            _convert_to_16k(path, out_path)
        sr, dur = SR, sf.info(str(out_path)).frames / SR
    row = {"path": str(out_path), "duration_s": round(dur, 3), "sr": sr}
    if speaker_id is not None:
        row["speaker_id"] = speaker_id
    return row


def scan_librispeech(root: Path, data_root: Path, prepared: Path, parts: list[str]):
    rows = []
    for part in parts:
        base = root / part
        if not base.exists():
            continue
        files = sorted(base.rglob("*.flac"))
        print(f"[manifest] {part}: {len(files)} files")
        for f in files:
            spk = f"ls_{f.relative_to(base).parts[0]}"
            rows.append(_entry(f, data_root, prepared, spk))
    return rows


def scan_vctk(root: Path, data_root: Path, prepared: Path):
    if not root.exists():
        return []
    files = sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav"))
    print(f"[manifest] vctk: {len(files)} files (48k -> 16k conversion on first run)")
    return [_entry(f, data_root, prepared, f"vctk_{f.stem.split('_')[0]}") for f in files]


def scan_noise(data_root: Path, prepared: Path):
    rows = []
    musan = data_root / "raw" / "musan"
    for sub in ("noise", "music"):
        base = musan / sub
        if base.exists():
            files = sorted(base.rglob("*.wav"))
            print(f"[manifest] musan/{sub}: {len(files)} files")
            rows += [_entry(f, data_root, prepared, None) for f in files]
    dns = data_root / "raw" / "dns_noise"
    if dns.exists():
        files = sorted(dns.rglob("*.wav"))
        print(f"[manifest] dns_noise: {len(files)} files (resampling 48k->16k on first run)")
        rows += [_entry(f, data_root, prepared, None) for f in files]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_root")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    prepared = data_root / "prepared"
    mdir = data_root / "manifests"
    ls_root = data_root / "raw" / "LibriSpeech"

    train_speech = scan_librispeech(
        ls_root, data_root, prepared, ["train-clean-100", "train-clean-360"]
    )
    train_speech += scan_vctk(data_root / "raw" / "vctk", data_root, prepared)
    val_speech = scan_librispeech(ls_root, data_root, prepared, ["dev-clean"])
    if not train_speech or not val_speech:
        raise SystemExit("LibriSpeech not found — run data/download.py librispeech first")

    noise = scan_noise(data_root, prepared)
    # file-disjoint deterministic 95/5 split by path hash
    def _is_val(r):
        return int(hashlib.md5(r["path"].encode()).hexdigest(), 16) % 20 == 0

    n_tr = [r for r in noise if not _is_val(r)]
    n_va = [r for r in noise if _is_val(r)]

    save_jsonl(mdir / "train_speech.jsonl", train_speech)
    save_jsonl(mdir / "val_speech.jsonl", val_speech)
    save_jsonl(mdir / "noise_train.jsonl", n_tr)
    save_jsonl(mdir / "noise_val.jsonl", n_va)

    spk = len({r["speaker_id"] for r in train_speech})
    print(f"\n[manifest] train speech: {len(train_speech)} utts / {spk} speakers")
    print(f"[manifest] val speech:   {len(val_speech)} utts")
    print(f"[manifest] noise:        {len(n_tr)} train / {len(n_va)} val")
    print(f"[manifest] written to {mdir}/")
    print("Next: python -m prism_tse.data.generate_rirs --data_root", data_root)


if __name__ == "__main__":
    main()
