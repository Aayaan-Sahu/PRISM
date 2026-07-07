"""Validation metrics.

  si_sdri    — SI-SDR improvement (est vs. mixture) on target-present samples
  leakage_db — output energy relative to input on target_absent samples
               (more negative = better muting of non-enrolled speakers)
  word_pres  — fraction of target-active frames whose output energy is within
               10 dB of the clean target's (proxy for "no words clipped")
"""

from __future__ import annotations

import argparse

import torch

from prism_tse.config import Config
from prism_tse.losses import si_sdr


def _frame_energy_db(wav: torch.Tensor, hop: int) -> torch.Tensor:
    """(B, N) -> (B, N//hop + 1) energies matching frame_active_labels framing."""
    B, N = wav.shape
    n_hops = N // hop
    he = wav[:, : n_hops * hop].reshape(B, n_hops, hop).pow(2).mean(-1)
    fe = torch.zeros(B, n_hops + 1, device=wav.device)
    fe[:, :-1] += he
    fe[:, 1:] += he
    return 10.0 * torch.log10(fe / 2.0 + 1e-10)


@torch.no_grad()
def run_validation(model, spk_enc, stft, loader, cfg: Config, device: str) -> dict:
    n_samples = cfg.segment_samples
    hop = cfg.audio.hop

    sisdri_sum, sisdri_n = 0.0, 0
    leak_sum, leak_n = 0.0, 0
    wp_hit, wp_total = 0, 0

    for batch in loader:
        mix = batch["mixture"].to(device)
        gt = batch["target"].to(device)
        active = batch["active"].to(device)
        emb = spk_enc.embed(batch["enroll"].to(device), batch["enroll_lens"])

        est_spec, _ = model(stft.stft(mix), emb)
        est = stft.istft(est_spec, n_samples).float()

        has_target = active.any(dim=1)
        if has_target.any():
            imp = si_sdr(est[has_target], gt[has_target]) - si_sdr(
                mix[has_target], gt[has_target]
            )
            sisdri_sum += float(imp.sum())
            sisdri_n += int(has_target.sum())

            e_db = _frame_energy_db(est[has_target], hop)
            g_db = _frame_energy_db(gt[has_target], hop)
            act = active[has_target]
            wp_hit += int(((e_db > g_db - 10.0) & act).sum())
            wp_total += int(act.sum())

        absent = torch.tensor(
            [s == "target_absent" for s in batch["scenario"]], device=device
        )
        if absent.any():
            e_out = est[absent].pow(2).mean(dim=1)
            e_in = mix[absent].pow(2).mean(dim=1)
            leak = 10.0 * torch.log10(e_out / (e_in + 1e-10) + 1e-10)
            leak_sum += float(leak.sum())
            leak_n += int(absent.sum())

    metrics = {}
    if sisdri_n:
        metrics["si_sdri"] = sisdri_sum / sisdri_n
    if leak_n:
        metrics["leakage_db"] = leak_sum / leak_n
    if wp_total:
        metrics["word_pres"] = wp_hit / wp_total
    return metrics


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from prism_tse.datasets.mixer import ValMixerDataset, collate
    from prism_tse.models.ecapa import get_speaker_encoder
    from prism_tse.models.separator import PrismTSE
    from prism_tse.models.stft import CausalStft

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n_val", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["cfg"])
    if args.data_root:
        cfg.paths.data_root = args.data_root

    model = PrismTSE(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    stft = CausalStft(cfg.audio.n_fft, cfg.audio.hop).to(device)
    spk_enc = get_speaker_encoder(cfg, device)
    loader = DataLoader(ValMixerDataset(cfg, args.n_val), batch_size=8,
                        num_workers=2, collate_fn=collate)
    metrics = run_validation(model, spk_enc, stft, loader, cfg, device)
    print(f"step {ck['step']}: " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
