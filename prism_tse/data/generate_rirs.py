"""Pre-generate a bank of room impulse responses.

    python -m prism_tse.data.generate_rirs --data_root data_root --n-rooms 500

Simulating rooms per training sample is too slow; instead we simulate once and
the mixer samples from the bank. Each room contributes several source
positions sharing one mic — the mixer assigns different sources of the SAME
room to target vs. interferers. openSLR RIRS_NOISES (if downloaded) is folded
into the same manifest for real-world diversity.

Output: {data_root}/rirs/synthetic/*.wav + {data_root}/manifests/rirs.jsonl
rows: {"room_id", "path", "rt60"}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from prism_tse.datasets.manifest import save_jsonl

SR = 16_000


def generate_synthetic(out_dir: Path, n_rooms: int, sources: int, seed: int) -> list[dict]:
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in range(n_rooms):
        dims = [float(rng.uniform(3, 10)), float(rng.uniform(3, 10)), float(rng.uniform(2.4, 4.0))]
        rt60 = float(rng.uniform(0.15, 0.7))
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, dims)
        except ValueError:
            continue
        room = pra.ShoeBox(dims, fs=SR, materials=pra.Material(e_abs),
                           max_order=min(int(max_order), 30))

        def rand_pos():
            return [float(rng.uniform(0.5, d - 0.5)) for d in dims]

        mic = rand_pos()
        room.add_microphone(mic)
        placed = 0
        for _ in range(sources * 4):
            if placed == sources:
                break
            p = rand_pos()
            if np.linalg.norm(np.array(p) - np.array(mic)) > 0.3:
                room.add_source(p)
                placed += 1
        if placed == 0:
            continue

        room.compute_rir()
        for s in range(placed):
            rir = np.asarray(room.rir[0][s], dtype=np.float32)
            onset = int(np.argmax(np.abs(rir)))
            rir = rir[max(0, onset - 8):][: int(0.6 * SR)]
            peak = float(np.max(np.abs(rir)))
            if peak < 1e-6:
                continue
            rir /= peak
            path = out_dir / f"room{r:05d}_src{s}.wav"
            sf.write(str(path), rir, SR)
            rows.append({"room_id": f"synth_{r:05d}", "path": str(path),
                         "rt60": round(rt60, 3)})
        if (r + 1) % 50 == 0:
            print(f"[rirs] {r + 1}/{n_rooms} rooms")
    return rows


def scan_openslr(raw: Path) -> list[dict]:
    """Fold RIRS_NOISES simulated RIRs in: each Room* folder = one room."""
    base = raw / "RIRS_NOISES" / "simulated_rirs"
    if not base.exists():
        return []
    rows = []
    for room_dir in sorted(base.glob("*/Room*")):
        for f in sorted(room_dir.glob("*.wav")):
            rows.append({"room_id": f"slr28_{room_dir.parent.name}_{room_dir.name}",
                         "path": str(f), "rt60": None})
    print(f"[rirs] openSLR RIRS_NOISES: {len(rows)} RIRs")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_root")
    ap.add_argument("--n-rooms", type=int, default=500)
    ap.add_argument("--sources-per-room", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    rows = []
    if not args.skip_synthetic:
        rows += generate_synthetic(data_root / "rirs" / "synthetic",
                                   args.n_rooms, args.sources_per_room, args.seed)
    rows += scan_openslr(data_root / "raw")
    if not rows:
        raise SystemExit("No RIRs generated or found")
    save_jsonl(data_root / "manifests" / "rirs.jsonl", rows)
    rooms = len({r["room_id"] for r in rows})
    print(f"[rirs] wrote {len(rows)} RIRs across {rooms} rooms -> manifests/rirs.jsonl")


if __name__ == "__main__":
    main()
