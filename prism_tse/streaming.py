"""Stateful streaming inference: one 16 ms hop in, one 16 ms hop out.

Matches CausalStft's offline convention exactly (see models/stft.py):
  * the initial zero input-carry == the offline left pad,
  * frame k is built from hops (k-1, k),
  * emission k = overlap-add tail + first half of synthesis frame k, which is
    the finished output for padded hop k. Offline output drops the pad hop, so
    process() discards the first emission and feeds one zero flush hop at the
    end — making offline forward and hop-by-hop streaming bit-comparable
    (tests/test_streaming_parity.py asserts allclose at 1e-4).

Total algorithmic latency = one window = 32 ms. Runs in float32: bf16 state
drift across thousands of LSTM steps hurts parity, and fp32 is far faster than
real time at 62.5 frames/s anyway.
"""

from __future__ import annotations

import torch

from prism_tse.models.stft import CausalStft


class StreamingSession:
    """One session per (audio stream x enrolled speaker). For N enrolled
    speakers, run N sessions on the same input hops and sum the outputs."""

    def __init__(self, model, stft: CausalStft, spk_emb: torch.Tensor, device: str = "cpu"):
        self.model = model.to(device).eval().float()
        self.stft = stft.to(device)
        self.device = device
        self.hop = stft.hop
        emb = spk_emb.to(device).float()
        self.spk_emb = emb.unsqueeze(0) if emb.dim() == 1 else emb  # (1, 192)
        self.reset()

    def reset(self) -> None:
        self.states = self.model.init_states(1, self.device, torch.float32)
        self.input_carry = torch.zeros(self.hop, device=self.device)
        self.ola_tail = torch.zeros(self.stft.n_fft - self.hop, device=self.device)

    @torch.no_grad()
    def process_hop(self, hop: torch.Tensor) -> torch.Tensor:
        """hop: (hop,) float32 mono samples -> (hop,) filtered samples.
        Output corresponds to the previous input hop (32 ms total latency)."""
        hop = hop.to(self.device).float()
        assert hop.shape == (self.hop,), f"expected ({self.hop},), got {tuple(hop.shape)}"

        frame = torch.cat([self.input_carry, hop]).unsqueeze(0)      # (1, n_fft)
        spec = self.stft.analyze_frame(frame).unsqueeze(1)           # (1, 1, F)
        est_spec, self.states = self.model(spec, self.spk_emb, self.states)
        y = self.stft.synthesize_frame(est_spec.squeeze(1)).squeeze(0)  # (n_fft,)

        out = self.ola_tail + y[: self.hop]
        self.ola_tail = y[self.hop :].clone()
        self.input_carry = hop
        return out

    @torch.no_grad()
    def process(self, wav: torch.Tensor) -> torch.Tensor:
        """Convenience offline-equivalent: (N,) -> (N,). Resets state first."""
        self.reset()
        wav = wav.to(self.device).float().flatten()
        n = wav.numel()
        pad = (-n) % self.hop
        if pad:
            wav = torch.cat([wav, torch.zeros(pad, device=self.device)])

        emissions = []
        for k in range(wav.numel() // self.hop):
            emissions.append(self.process_hop(wav[k * self.hop : (k + 1) * self.hop]))
        emissions.append(self.process_hop(torch.zeros(self.hop, device=self.device)))

        # emission 0 is the pad region (offline drops it); trim to input length
        return torch.cat(emissions[1:])[:n]
