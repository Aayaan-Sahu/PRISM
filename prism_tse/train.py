"""PRISM-TSE training loop (single GPU, plain PyTorch).

    python -m prism_tse.train --config prism_tse/configs/default.yaml \
        --data_root /path/to/data_root --run_dir runs/v1
    # resume:
    python -m prism_tse.train --config ... --resume auto

bf16 autocast on CUDA (no GradScaler — bf16 has fp32-range exponent). Loss
math and the STFT/mask path run in fp32. Checkpoints: latest.pt / best.pt /
step_XXXXXXX.pt (pruned to keep_last). Resume restores model/opt/sched/step;
the data stream deliberately reseeds (fresh random mixtures, not bit-exact).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prism_tse.config import Config
from prism_tse.datasets.mixer import TrainMixer, ValMixerDataset, collate
from prism_tse.losses import total_loss
from prism_tse.models.ecapa import get_speaker_encoder
from prism_tse.models.separator import PrismTSE, count_params
from prism_tse.models.stft import CausalStft
from prism_tse.validate import run_validation


def make_scheduler(opt, warmup: int, total: int):
    def lam(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        t = (step - warmup) / max(total - warmup, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))), 0.02)

    return torch.optim.lr_scheduler.LambdaLR(opt, lam)


def save_ckpt(path: Path, step: int, model, opt, sched, cfg: Config, best: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "cfg": cfg.to_dict(),
            "best_val_sisdri": best,
        },
        path,
    )


def prune_ckpts(ckpt_dir: Path, keep_last: int):
    steps = sorted(ckpt_dir.glob("step_*.pt"))
    for old in steps[:-keep_last]:
        old.unlink(missing_ok=True)


def get_writer(run_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(run_dir / "tb"))
    except ImportError:
        print("[train] tensorboard not installed — scalar logging to stdout only")
        return None


def main(cfg: Config, resume: str | None = None, device: str | None = None) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.train.seed)

    run_dir = Path(cfg.paths.run_dir)
    ckpt_dir = run_dir / "ckpt"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    stft = CausalStft(cfg.audio.n_fft, cfg.audio.hop).to(device)
    model = PrismTSE(cfg).to(device)
    spk_enc = get_speaker_encoder(cfg, device)
    print(f"[train] device={device} params={count_params(model)/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = make_scheduler(opt, cfg.train.warmup, cfg.train.steps)

    start_step, best = 0, float("-inf")
    if resume:
        path = ckpt_dir / "latest.pt" if resume == "auto" else Path(resume)
        if path.exists():
            ck = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_step = ck["step"]
            best = ck.get("best_val_sisdri", float("-inf"))
            print(f"[train] resumed from {path} at step {start_step}")
        elif resume != "auto":
            raise FileNotFoundError(path)

    train_loader = DataLoader(
        TrainMixer(cfg, seed_salt=start_step),
        batch_size=cfg.train.batch,
        num_workers=cfg.train.workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=4 if cfg.train.workers > 0 else None,
    )
    val_loader = DataLoader(
        ValMixerDataset(cfg),
        batch_size=min(cfg.train.batch, 16),
        num_workers=min(cfg.train.workers, 2),
        collate_fn=collate,
    )

    writer = get_writer(run_dir)
    amp = cfg.train.amp and device == "cuda"
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
    )

    n_samples = cfg.segment_samples
    model.train()
    it = iter(train_loader)
    t0, seen = time.time(), 0
    last_logs: dict = {}

    for step in range(start_step + 1, cfg.train.steps + 1):
        batch = next(it)
        mix = batch["mixture"].to(device, non_blocking=True)
        gt = batch["target"].to(device, non_blocking=True)
        enroll = batch["enroll"].to(device, non_blocking=True)
        active = batch["active"].to(device, non_blocking=True)

        emb = spk_enc.embed(enroll, batch["enroll_lens"])       # fp32, no_grad
        spec = stft.stft(mix)                                   # fp32 complex
        ref_spec = stft.stft(gt)
        with autocast:
            est_spec, _ = model(spec, emb)                      # mask applied fp32
        est_wav = stft.istft(est_spec, n_samples)

        loss, logs = total_loss(est_wav, est_spec, gt, ref_spec, active,
                                cfg.train, cfg.audio.power)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip)
        opt.step()
        sched.step()
        last_logs = logs
        seen += mix.shape[0]

        if step % cfg.train.log_every == 0:
            sps = seen / max(time.time() - t0, 1e-6)
            t0, seen = time.time(), 0
            lr = sched.get_last_lr()[0]
            msg = " ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in logs.items())
            print(f"[train] step {step}/{cfg.train.steps} {msg} lr={lr:.2e} "
                  f"({sps:.1f} samp/s)")
            if writer:
                for k, v in logs.items():
                    writer.add_scalar(k, v, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/samples_per_s", sps, step)

        if step % cfg.train.val_every == 0 or step == cfg.train.steps:
            model.eval()
            metrics = run_validation(model, spk_enc, stft, val_loader, cfg, device)
            model.train()
            print(f"[val]   step {step} " +
                  " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, step)
            if metrics.get("si_sdri", float("-inf")) > best:
                best = metrics["si_sdri"]
                save_ckpt(ckpt_dir / "best.pt", step, model, opt, sched, cfg, best)

        if step % cfg.train.ckpt_every == 0 or step == cfg.train.steps:
            save_ckpt(ckpt_dir / f"step_{step:07d}.pt", step, model, opt, sched, cfg, best)
            shutil.copyfile(ckpt_dir / f"step_{step:07d}.pt", ckpt_dir / "latest.pt")
            prune_ckpts(ckpt_dir, cfg.train.keep_last)

    if writer:
        writer.close()
    return {"step": cfg.train.steps, "loss": last_logs.get("loss/total", float("nan")),
            "best_val_sisdri": best}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="prism_tse/configs/default.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--resume", default=None, help='"auto" or a checkpoint path')
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.paths.data_root = args.data_root
    if args.run_dir:
        cfg.paths.run_dir = args.run_dir
    main(cfg, resume=args.resume, device=args.device)
