"""Mixer sanity: shapes, scenario frequencies, enrollment pairing, labels."""

import numpy as np
import torch

from prism_tse.datasets.mixer import (
    ValMixerDataset,
    build_sample,
    collate,
    load_mixer_inputs,
)

N_SAMPLES = 200


def test_build_samples(synth_corpus):
    cfg = synth_corpus
    idx, noise, rooms = load_mixer_inputs(cfg, "train")
    rng = np.random.default_rng(42)

    n_seg = cfg.segment_samples
    n_frames = n_seg // cfg.audio.hop + 1
    counts: dict[str, int] = {}

    for _ in range(N_SAMPLES):
        s = build_sample(rng, idx, noise, rooms, cfg)
        counts[s["scenario"]] = counts.get(s["scenario"], 0) + 1

        assert s["mixture"].shape == (n_seg,)
        assert s["target"].shape == (n_seg,)
        assert s["active"].shape == (n_frames,)
        assert torch.isfinite(s["mixture"]).all()
        assert float(s["mixture"].abs().max()) <= 1.0 + 1e-4

        if s["scenario"] == "target_absent":
            assert not s["active"].any()
            assert float(s["target"].abs().max()) == 0.0
        else:
            assert s["active"].any(), "target present but no active frames"

        assert 0.0 < s["enroll_len"] <= 1.0
        assert float(s["enroll"].abs().max()) > 0.0

    # every scenario appears, and frequencies are in the right ballpark
    probs = cfg.mixer.scenario_probs
    assert set(counts) == set(probs)
    for name, p in probs.items():
        assert abs(counts[name] / N_SAMPLES - p) < 0.15, (name, counts)


def test_val_dataset_deterministic(synth_corpus):
    cfg = synth_corpus
    ds1 = ValMixerDataset(cfg, n_items=4)
    ds2 = ValMixerDataset(cfg, n_items=4)
    for i in range(4):
        a, b = ds1[i], ds2[i]
        assert torch.equal(a["mixture"], b["mixture"])
        assert torch.equal(a["target"], b["target"])
        assert a["scenario"] == b["scenario"]


def test_collate(synth_corpus):
    cfg = synth_corpus
    ds = ValMixerDataset(cfg, n_items=3)
    batch = collate([ds[i] for i in range(3)])
    assert batch["mixture"].shape[0] == 3
    assert batch["enroll_lens"].shape == (3,)
    assert len(batch["scenario"]) == 3
