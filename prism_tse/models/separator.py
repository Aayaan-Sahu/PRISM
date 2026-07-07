"""PRISM-TSE separator: causal, speaker-conditioned complex masking model.

    compressed |STFT| -> Linear(F->H) -> freq convs -> 3 x [FiLM -> uniLSTM]
                      -> Linear(H -> 2F) complex ratio mask -> Y = X * M

Everything is frame-local or strictly left-to-right (uni-directional LSTM), so
the same forward() serves offline training (T frames at once, states=None) and
streaming inference (T=1, states threaded between calls). ~2M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from prism_tse.config import Config


class SpeakerMLP(nn.Module):
    """Frozen-ECAPA embedding (192) -> conditioning vector e (spk_dim)."""

    def __init__(self, emb_dim: int, spk_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, spk_dim),
            nn.LayerNorm(spk_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:  # (B, emb_dim) -> (B, spk_dim)
        return self.net(emb)


class FiLM(nn.Module):
    """Feature-wise linear modulation: x * (1 + gamma(e)) + beta(e).

    gamma/beta are broadcast over time, so the speaker embedding re-tunes each
    feature channel uniformly. (1 + gamma) keeps the block an identity at init.
    """

    def __init__(self, spk_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(spk_dim, feat_dim)
        self.to_beta = nn.Linear(spk_dim, feat_dim)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feat), e: (B, spk_dim)
        return x * (1.0 + self.to_gamma(e)).unsqueeze(1) + self.to_beta(e).unsqueeze(1)


class FreqConvBlock(nn.Module):
    """Convolution across the FEATURE axis within each frame (never across time,
    which would peek into the future and break causality)."""

    def __init__(self, ch: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(1, ch, kernel, padding=pad)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv1d(ch, 1, kernel, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, H)
        B, T, H = x.shape
        y = x.reshape(B * T, 1, H)
        y = self.conv2(self.act(self.conv1(y)))
        return x + y.reshape(B, T, H)


class PrismTSE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.power = cfg.audio.power
        self.f_bins = cfg.audio.n_fft // 2 + 1
        self.hidden = m.hidden
        self.n_lstm = m.n_lstm

        self.in_proj = nn.Linear(self.f_bins, m.hidden)
        self.freq_blocks = nn.ModuleList(
            FreqConvBlock(m.freq_conv_ch, m.freq_conv_kernel)
            for _ in range(m.freq_conv_blocks)
        )
        self.spk_mlp = SpeakerMLP(m.emb_dim, m.spk_dim)
        self.films = nn.ModuleList(FiLM(m.spk_dim, m.hidden) for _ in range(m.n_lstm))
        self.lstms = nn.ModuleList(
            nn.LSTM(m.hidden, m.hidden, batch_first=True) for _ in range(m.n_lstm)
        )
        self.mask_head = nn.Linear(m.hidden, 2 * self.f_bins)

    def init_states(self, batch: int, device, dtype=torch.float32):
        return [
            (
                torch.zeros(1, batch, self.hidden, device=device, dtype=dtype),
                torch.zeros(1, batch, self.hidden, device=device, dtype=dtype),
            )
            for _ in range(self.n_lstm)
        ]

    def forward(
        self,
        spec: torch.Tensor,           # (B, T, F) complex — mixture STFT
        spk_emb: torch.Tensor,        # (B, emb_dim) — frozen ECAPA embedding
        states: list | None = None,   # LSTM states for streaming; None for training
    ):
        B, T, Fb = spec.shape

        # Features stay real; masking at the end happens in fp32 regardless of AMP.
        mag = spec.abs().float().clamp_min(1e-5).pow(self.power)  # (B, T, F)
        x = self.in_proj(mag)
        for blk in self.freq_blocks:
            x = blk(x)

        e = self.spk_mlp(spk_emb)

        new_states = []
        for i in range(self.n_lstm):
            x = self.films[i](x, e)
            x, st = self.lstms[i](x, states[i] if states is not None else None)
            new_states.append(st)

        # Complex ratio mask, magnitude bounded by tanh (attenuate-only).
        m = self.mask_head(x).float().view(B, T, 2, Fb)
        r = torch.linalg.vector_norm(m, dim=2)                    # (B, T, F)
        scale = torch.tanh(r) / (r + 1e-8)
        mask = torch.complex(m[:, :, 0] * scale, m[:, :, 1] * scale)
        return spec * mask, new_states


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = Config()
    model = PrismTSE(cfg)
    n = count_params(model)
    print(f"PrismTSE parameters: {n / 1e6:.2f} M ({n:,})")
