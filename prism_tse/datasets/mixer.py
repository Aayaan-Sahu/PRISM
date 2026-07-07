"""On-the-fly mixture synthesis.

Every training sample is manufactured from clean corpora:
  target speech (+ its early-reverb copy as ground truth) + interfering
  speakers + noise, each reverberated with per-source RIRs from one room,
  followed by mixture-level channel simulation (EQ / tilt / clipping / LUFS).

TrainMixer is an infinite IterableDataset (per-worker RNG); ValMixerDataset is
a map-style Dataset whose items are pure functions of (val_seed, index) —
deterministic across runs and worker counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from prism_tse.config import Config
from prism_tse.datasets import augment as A
from prism_tse.datasets.manifest import SpeakerIndex, load_jsonl, load_rir_rooms

SCENARIOS = {
    # name: (n_interferers, has_noise, target_present)
    "target_alone":         (0, False, True),
    "target_noise":         (0, True,  True),
    "target_1interf":       (1, False, True),
    "target_1interf_noise": (1, True,  True),
    "target_2interf_noise": (2, True,  True),
    "target_absent":        (None, True, False),  # 1-2 interferers, chosen at runtime
}


# --------------------------------------------------------------------------
# Core sample builder
# --------------------------------------------------------------------------
def _place(rng, canvas_len: int, seg: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Drop `seg` at a random onset inside a zero canvas."""
    canvas = np.zeros(canvas_len, dtype=np.float32)
    onset = int(rng.integers(0, canvas_len - len(seg) + 1))
    canvas[onset : onset + len(seg)] = seg
    return canvas, onset, onset + len(seg)


def frame_active_labels(gt: np.ndarray, hop: int, thresh_db: float) -> np.ndarray:
    """(N,) ground truth -> (N//hop + 1,) bool frame labels matching the
    CausalStft framing (frame k spans hops k-1 and k)."""
    n_hops = len(gt) // hop
    if n_hops == 0:
        return np.zeros(1, dtype=bool)
    he = (gt[: n_hops * hop].reshape(n_hops, hop) ** 2).mean(axis=1)
    peak = float(he.max())
    if peak < 1e-10:  # silence / target absent
        return np.zeros(n_hops + 1, dtype=bool)
    hop_active = 10.0 * np.log10(he + 1e-12) > 10.0 * np.log10(peak) + thresh_db
    frames = np.zeros(n_hops + 1, dtype=bool)
    frames[:-1] |= hop_active          # frame k sees hop k
    frames[1:] |= hop_active           # frame k+1 sees hop k
    return frames


def build_sample(
    rng: np.random.Generator,
    spk_index: SpeakerIndex,
    noise_entries: list[dict],
    rir_rooms: list[list[str]],
    cfg: Config,
) -> dict:
    mx, au = cfg.mixer, cfg.audio
    sr = au.sr
    n = cfg.segment_samples

    names = list(mx.scenario_probs)
    probs = np.array([mx.scenario_probs[k] for k in names], dtype=np.float64)
    scenario = names[int(rng.choice(len(names), p=probs / probs.sum()))]
    n_interf, has_noise, target_present = SCENARIOS[scenario]
    if n_interf is None:
        n_interf = int(rng.integers(1, 3))
    if has_noise and not noise_entries:
        has_noise = False

    # -- room / RIRs --------------------------------------------------------
    use_reverb = bool(rir_rooms) and rng.random() < mx.reverb_prob
    room = rir_rooms[int(rng.integers(len(rir_rooms)))] if use_reverb else None

    def source_rir(i: int) -> np.ndarray | None:
        if room is None:
            return None
        return A.load_rir(room[i % len(room)], sr)

    # -- target + enrollment (always sampled; embedding needed even if absent)
    utt_mix, utt_enroll, spk = spk_index.sample_target(rng)

    mixture = np.zeros(n, dtype=np.float32)
    gt = np.zeros(n, dtype=np.float32)
    speech_for_snr = np.zeros(n, dtype=np.float32)

    if target_present:
        dur = float(rng.uniform(mx.min_target_s, mx.segment_s))
        seg = A.load_segment(utt_mix["path"], int(dur * sr), rng, sr)
        rir = source_rir(0)
        if rir is not None:
            wet = A.apply_rir(seg, rir)
            dry_gt = A.apply_rir(seg, A.early_rir(rir, sr, mx.early_ms))
        else:
            wet, dry_gt = seg, seg
        canvas, onset, _ = _place(rng, n, wet)
        gt[onset : onset + len(dry_gt)] = dry_gt
        mixture += canvas
        speech_for_snr += canvas

    # -- interferers ---------------------------------------------------------
    for k in range(n_interf):
        e = spk_index.sample_interferer(rng, exclude_speaker=spk)
        dur = float(rng.uniform(mx.min_interf_s, mx.segment_s))
        seg = A.load_segment(e["path"], int(dur * sr), rng, sr)
        rir = source_rir(k + 1)
        if rir is not None:
            seg = A.apply_rir(seg, rir)
        canvas, _, _ = _place(rng, n, seg)
        if target_present:
            g = A.gain_for_ratio(gt, canvas, float(rng.uniform(*mx.sir_db)))
        else:
            g = 1.0 / (A.rms(canvas) + A.EPS) * 0.05  # ~-26 dBFS nominal
        mixture += g * canvas
        speech_for_snr += g * canvas

    # -- noise ---------------------------------------------------------------
    if has_noise:
        ne = noise_entries[int(rng.integers(len(noise_entries)))]
        noise = A.load_segment(ne["path"], n, rng, sr)
        ref = speech_for_snr if A.rms(speech_for_snr) > 1e-6 else mixture
        if A.rms(ref) < 1e-6:
            ref = noise
        mixture += A.gain_for_ratio(ref, noise, float(rng.uniform(*mx.snr_db))) * noise

    # -- mixture-level channel simulation -------------------------------------
    # EQ/tilt/clip are applied to the MIXTURE only (the model learns to undo
    # them); the clip stage's equivalent linear gain and the final LUFS gain
    # are also applied to the ground truth so levels stay aligned.
    mixture = A.random_eq(mixture, rng, sr)
    mixture = A.spectral_tilt(mixture, rng)
    mixture, clip_gain = A.maybe_clip(mixture, rng, mx.clip_prob)
    gt *= clip_gain

    g = A.lufs_gain(mixture, float(rng.uniform(*mx.lufs)), sr)
    g = float(np.clip(g, 0.0, 1e4))
    mixture *= g
    gt *= g
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.99:  # rare hot mixtures: rescale both, keep alignment
        mixture *= 0.99 / peak
        gt *= 0.99 / peak

    active = frame_active_labels(gt, au.hop, mx.active_thresh_db)

    # -- enrollment audio ------------------------------------------------------
    enroll_n = int(mx.enroll_crop_s * sr)
    e_avail = min(enroll_n, int(utt_enroll["duration_s"] * sr))
    enroll = A.load_segment(utt_enroll["path"], e_avail, rng, sr)
    if rng.random() < mx.enroll_degrade_prob:
        if rir_rooms:
            r_room = rir_rooms[int(rng.integers(len(rir_rooms)))]
            enroll = A.apply_rir(enroll, A.load_rir(r_room[0], sr))
        if noise_entries:
            ne = noise_entries[int(rng.integers(len(noise_entries)))]
            nz = A.load_segment(ne["path"], len(enroll), rng, sr)
            enroll += A.gain_for_ratio(enroll, nz, float(rng.uniform(*mx.enroll_snr_db))) * nz
    epk = float(np.max(np.abs(enroll)))
    if epk > A.EPS:
        enroll = enroll / epk * 0.7
    enroll_len = len(enroll) / enroll_n
    if len(enroll) < enroll_n:
        enroll = np.pad(enroll, (0, enroll_n - len(enroll)))

    return {
        "mixture": torch.from_numpy(mixture),
        "target": torch.from_numpy(gt),
        "enroll": torch.from_numpy(enroll.astype(np.float32)),
        "enroll_len": float(enroll_len),
        "active": torch.from_numpy(active),
        "scenario": scenario,
    }


def collate(batch: list[dict]) -> dict:
    return {
        "mixture": torch.stack([b["mixture"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "enroll": torch.stack([b["enroll"] for b in batch]),
        "enroll_lens": torch.tensor([b["enroll_len"] for b in batch]),
        "active": torch.stack([b["active"] for b in batch]),
        "scenario": [b["scenario"] for b in batch],
    }


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def load_mixer_inputs(cfg: Config, split: str):
    """split: 'train' | 'val'. Reads manifests under {data_root}/manifests."""
    mdir = cfg.manifests_dir
    speech = load_jsonl(mdir / f"{split}_speech.jsonl")
    noise_path = mdir / f"noise_{split}.jsonl"
    noise = load_jsonl(noise_path) if noise_path.exists() else []
    rooms = load_rir_rooms(mdir / "rirs.jsonl")
    idx = SpeakerIndex(speech, cfg.mixer.min_utt_s, cfg.mixer.min_interf_s)
    return idx, noise, rooms


class TrainMixer(IterableDataset):
    def __init__(self, cfg: Config, seed_salt: int = 0):
        super().__init__()
        self.cfg = cfg
        self.seed_salt = seed_salt
        self.spk_index, self.noise, self.rooms = load_mixer_inputs(cfg, "train")

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = np.random.default_rng(
            [self.cfg.train.seed, worker_id, self.seed_salt]
        )
        failures = 0
        while True:
            try:
                yield build_sample(rng, self.spk_index, self.noise, self.rooms, self.cfg)
                failures = 0
            except Exception as e:  # corrupt file etc. — skip, but not silently forever
                failures += 1
                if failures in (1, 10) or failures % 100 == 0:
                    print(f"[mixer] sample build failed x{failures}: {e!r}")
                if failures > 1000:
                    raise


class ValMixerDataset(Dataset):
    """Deterministic: item i depends only on (val_seed, i)."""

    VAL_SEED = 7_777

    def __init__(self, cfg: Config, n_items: int | None = None):
        self.cfg = cfg
        self.n = n_items if n_items is not None else cfg.train.n_val
        self.spk_index, self.noise, self.rooms = load_mixer_inputs(cfg, "val")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng([self.VAL_SEED, i])
        return build_sample(rng, self.spk_index, self.noise, self.rooms, self.cfg)


# --------------------------------------------------------------------------
# Debug CLI: render examples to disk for listening checks
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import soundfile as sf

    ap = argparse.ArgumentParser(description="Dump example mixtures for listening")
    ap.add_argument("--config", default="prism_tse/configs/default.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default="mixer_examples")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.paths.data_root = args.data_root
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    idx, noise, rooms = load_mixer_inputs(cfg, args.split)
    rng = np.random.default_rng(0)
    for i in range(args.n):
        s = build_sample(rng, idx, noise, rooms, cfg)
        stem = out / f"{i:02d}_{s['scenario']}"
        sf.write(f"{stem}_mix.wav", s["mixture"].numpy(), cfg.audio.sr)
        sf.write(f"{stem}_gt.wav", s["target"].numpy(), cfg.audio.sr)
        sf.write(f"{stem}_enroll.wav", s["enroll"].numpy(), cfg.audio.sr)
        print(f"{stem}: active {int(s['active'].sum())}/{len(s['active'])} frames")
    print(f"Wrote {args.n} examples to {out}/")
