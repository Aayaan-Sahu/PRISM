"""Manifest I/O and the speaker index used for enrollment pairing."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class SpeakerIndex:
    """Groups speech-manifest entries by speaker.

    Target/enrollment sampling needs speakers with >= 2 utterances of at least
    `min_utt_s`, so the enrollment utterance can always differ from the
    in-mixture utterance. Anything >= `min_interf_s` is usable as interference.
    """

    def __init__(self, entries: list[dict], min_utt_s: float, min_interf_s: float = 1.0):
        by_spk: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            if e["duration_s"] >= min_utt_s:
                by_spk[str(e["speaker_id"])].append(e)

        self.by_spk = {s: utts for s, utts in by_spk.items() if len(utts) >= 2}
        self.speakers = sorted(self.by_spk)
        self.interferers = [e for e in entries if e["duration_s"] >= min_interf_s]
        if not self.speakers:
            raise ValueError(
                "No eligible target speakers (need >= 2 utterances >= "
                f"{min_utt_s}s each). Check your manifests."
            )

    def sample_target(self, rng) -> tuple[dict, dict, str]:
        """Returns (mixture_utterance, enrollment_utterance, speaker_id) —
        two DIFFERENT utterances of the same speaker."""
        spk = self.speakers[int(rng.integers(len(self.speakers)))]
        utts = self.by_spk[spk]
        i, j = rng.choice(len(utts), size=2, replace=False)
        return utts[int(i)], utts[int(j)], spk

    def sample_interferer(self, rng, exclude_speaker: str) -> dict:
        for _ in range(50):
            e = self.interferers[int(rng.integers(len(self.interferers)))]
            if str(e["speaker_id"]) != exclude_speaker:
                return e
        raise RuntimeError("Could not sample an interferer from another speaker")


def load_rir_rooms(path: str | Path) -> list[list[str]]:
    """rirs.jsonl -> list of rooms, each a list of RIR wav paths (>=1 source).
    Returns [] if the manifest is missing (mixer then disables reverb)."""
    path = Path(path)
    if not path.exists():
        return []
    rooms: dict[str, list[str]] = defaultdict(list)
    for row in load_jsonl(path):
        rooms[str(row["room_id"])].append(row["path"])
    return [sorted(v) for _, v in sorted(rooms.items())]
