"""Frozen ECAPA-TDNN speaker encoder (mirrors understanding_enroll.py in the
main repo, so embeddings stay compatible with the existing enrollment flow).

Cluster usage: compute nodes often have no internet. Run
    python -m prism_tse.models.ecapa --predownload --savedir <dir>
once on a login node; afterwards FrozenEcapa loads purely from the local dir
(set HF_HUB_OFFLINE=1 in the job script for belt-and-braces).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

HF_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class FrozenEcapa:
    def __init__(self, savedir: str, device: str):
        from speechbrain.inference.speaker import EncoderClassifier

        local = Path(savedir)
        source = str(local) if (local / "hyperparams.yaml").exists() else HF_SOURCE
        self.model = EncoderClassifier.from_hparams(
            source=source, savedir=savedir, run_opts={"device": device}
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        """wav: (B, N) float32 in [-1,1]; wav_lens: (B,) relative lengths in (0,1].
        Returns L2-normalized (B, 192) float32."""
        emb = self.model.encode_batch(
            wav.to(self.device).float(), wav_lens=wav_lens.to(self.device).float()
        )
        emb = emb.squeeze(1).float()  # (B, 1, 192) -> (B, 192)
        return F.normalize(emb, dim=-1)


class FakeEcapa:
    """Deterministic, network-free stand-in for tests: projects simple frame
    energy statistics through a fixed random matrix. Same signature/shapes."""

    def __init__(self, device: str, emb_dim: int = 192, n_feats: int = 64, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(n_feats, emb_dim, generator=g).to(device)
        self.n_feats = n_feats
        self.device = device

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        wav = wav.to(self.device).float()
        B, N = wav.shape
        n = N - (N % self.n_feats)
        chunks = wav[:, :n].reshape(B, self.n_feats, -1)
        feats = torch.log10(chunks.pow(2).mean(-1) + 1e-8)  # (B, n_feats)
        return F.normalize(feats @ self.proj, dim=-1)


def get_speaker_encoder(cfg, device: str):
    if cfg.train.fake_ecapa:
        return FakeEcapa(device, emb_dim=cfg.model.emb_dim)
    return FrozenEcapa(cfg.ecapa_dir, device)


def predownload(savedir: str) -> None:
    from speechbrain.inference.speaker import EncoderClassifier

    EncoderClassifier.from_hparams(source=HF_SOURCE, savedir=savedir,
                                   run_opts={"device": "cpu"})
    print(f"[ecapa] model cached in {savedir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predownload", action="store_true")
    ap.add_argument("--savedir", required=True)
    args = ap.parse_args()
    if args.predownload:
        predownload(args.savedir)
