"""Causal STFT with exact streaming parity.

Do NOT replace this with torch.stft/istft: torch.stft(center=True) reflect-pads
by n_fft//2 (non-causal) and torch.istft divides by a window-envelope that
complicates streaming overlap-add. Here we own the framing:

  * analysis + synthesis window: sqrt(periodic Hann). At hop = n_fft/2 the
    product window (= Hann) satisfies COLA with sum exactly 1, so synthesis is
    a plain overlap-add with no normalization denominator.
  * convention: the signal is zero-padded by (n_fft - hop) samples on BOTH
    sides. Left pad == the streaming session's initial zero input-carry, so
    offline frame k is identical to the frame built from hops (k-1, k) with
    h_{-1} = 0. Right pad == the streaming flush hop.

For an input of N samples (N % hop == 0) this yields T = N//hop + 1 frames and
istft() reconstructs all N samples exactly (COLA holds over the whole range).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalStft(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 256):
        super().__init__()
        if n_fft != 2 * hop:
            raise ValueError("CausalStft requires hop == n_fft // 2 (sqrt-Hann COLA)")
        self.n_fft = n_fft
        self.hop = hop
        self.pad = n_fft - hop  # 256
        self.f_bins = n_fft // 2 + 1
        window = torch.sqrt(torch.hann_window(n_fft, periodic=True))
        self.register_buffer("window", window, persistent=False)

    def num_frames(self, n_samples: int) -> int:
        assert n_samples % self.hop == 0, "input length must be a multiple of hop"
        return n_samples // self.hop + 1

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, N) float -> (B, T, F) complex, T = N//hop + 1."""
        assert wav.dim() == 2 and wav.shape[1] % self.hop == 0
        x = F.pad(wav, (self.pad, self.pad))
        frames = x.unfold(1, self.n_fft, self.hop)          # (B, T, n_fft)
        return torch.fft.rfft(frames * self.window, dim=-1)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """(B, T, F) complex -> (B, length) float. Exact inverse of stft()."""
        frames = torch.fft.irfft(spec, n=self.n_fft, dim=-1) * self.window  # (B, T, n_fft)
        B, T, _ = frames.shape
        total = (T - 1) * self.hop + self.n_fft
        out = F.fold(
            frames.transpose(1, 2),                          # (B, n_fft, T)
            output_size=(1, total),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop),
        ).view(B, total)
        return out[:, self.pad : self.pad + length]

    # -- single-frame helpers used by streaming.py --------------------------
    def analyze_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """(B, n_fft) windowed-and-transformed -> (B, F) complex."""
        return torch.fft.rfft(frame * self.window, dim=-1)

    def synthesize_frame(self, spec: torch.Tensor) -> torch.Tensor:
        """(B, F) complex -> (B, n_fft) time-domain synthesis frame."""
        return torch.fft.irfft(spec, n=self.n_fft, dim=-1) * self.window
