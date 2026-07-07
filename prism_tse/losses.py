"""Loss functions. All math in float32 (callers cast bf16 outputs up first —
PrismTSE already returns fp32 spectra).

  target-active rows :  -SI-SDR(waveform)  +  spec_w * asymmetric spectral loss
  inactive frames    :  absent_w * output spectral energy  (silence supervision)

The asymmetric term is the anti-word-clipping mechanism: over-suppression of
the target (output missing energy the clean reference has) is penalized
spec_alpha x harder than under-suppression (residual leakage).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR per sample. est/ref: (B, N) -> (B,) in dB.
    Callers must exclude silent references (undefined)."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = (est * ref).sum(dim=-1, keepdim=True)
    s = dot / (ref.pow(2).sum(dim=-1, keepdim=True) + eps) * ref
    e = est - s
    return 10.0 * torch.log10(
        (s.pow(2).sum(dim=-1) + eps) / (e.pow(2).sum(dim=-1) + eps)
    )


def _safe_mag(spec: torch.Tensor, power: float, eps: float = 1e-10) -> torch.Tensor:
    """|spec|^power with a NaN-safe gradient. torch's complex abs() has a NaN
    gradient at exactly 0, and silent mixture regions (e.g. target_alone
    scenarios with no noise) produce exactly-zero STFT cells."""
    energy = spec.real.float().pow(2) + spec.imag.float().pow(2)
    return (energy + eps).pow(power / 2.0)


def asym_spec_loss(
    est_spec: torch.Tensor,   # (B, T, F) complex
    ref_spec: torch.Tensor,   # (B, T, F) complex
    active: torch.Tensor,     # (B, T) bool — frames where the target is speaking
    alpha: float,
    power: float,
) -> torch.Tensor:
    d = _safe_mag(ref_spec, power) - _safe_mag(est_spec, power)
    cell = F.relu(d).pow(2) * alpha + F.relu(-d).pow(2)   # (B, T, F)
    mask = active.float().unsqueeze(-1)
    return (cell * mask).sum() / mask.sum().clamp_min(1.0) / cell.shape[-1]


def absent_energy_loss(est_spec: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Mean output power over frames where the target is NOT speaking — covers
    both target_absent samples and pre-onset/post-offset regions."""
    inactive = (~active).float().unsqueeze(-1)
    energy = est_spec.real.float().pow(2) + est_spec.imag.float().pow(2)
    return (energy * inactive).sum() / inactive.sum().clamp_min(1.0) / energy.shape[-1]


def total_loss(
    est_wav: torch.Tensor,    # (B, N) fp32
    est_spec: torch.Tensor,   # (B, T, F) complex fp32
    gt_wav: torch.Tensor,     # (B, N)
    ref_spec: torch.Tensor,   # (B, T, F) complex
    active: torch.Tensor,     # (B, T) bool
    cfg_train,
    power: float,
):
    logs: dict[str, float] = {}
    has_target = active.any(dim=1)

    loss = est_wav.new_zeros(())
    if has_target.any():
        l_sisdr = -si_sdr(est_wav[has_target].float(), gt_wav[has_target].float()).mean()
        loss = loss + cfg_train.sisdr_w * l_sisdr
        logs["loss/sisdr"] = float(l_sisdr.detach())

    l_spec = asym_spec_loss(est_spec, ref_spec, active, cfg_train.spec_alpha, power)
    l_abs = absent_energy_loss(est_spec, active)
    loss = loss + cfg_train.spec_w * l_spec + cfg_train.absent_w * l_abs

    logs["loss/spec"] = float(l_spec.detach())
    logs["loss/absent"] = float(l_abs.detach())
    logs["loss/total"] = float(loss.detach())
    return loss, logs
