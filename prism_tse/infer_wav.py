"""Run a checkpoint on a wav file for listening tests.

    python -m prism_tse.infer_wav --ckpt runs/v1/ckpt/best.pt \
        --mix mixture.wav --enroll alice.wav --enroll bob.wav \
        --out filtered.wav [--streaming]

Multiple --enroll flags = multi-speaker mode: the model runs once per enrolled
voice and the outputs are summed (the PRISM deployment pattern).
"""

from __future__ import annotations

import argparse

import soundfile as sf
import torch
import torchaudio

from prism_tse.config import Config
from prism_tse.models.ecapa import get_speaker_encoder
from prism_tse.models.separator import PrismTSE
from prism_tse.models.stft import CausalStft
from prism_tse.streaming import StreamingSession


def load_16k_mono(path: str, sr: int = 16000) -> torch.Tensor:
    wav, file_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    return wav.squeeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--enroll", action="append", required=True,
                    help="enrollment wav; repeat for multiple speakers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--streaming", action="store_true",
                    help="use hop-by-hop StreamingSession instead of one forward")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["cfg"])

    model = PrismTSE(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    stft = CausalStft(cfg.audio.n_fft, cfg.audio.hop).to(device)
    spk_enc = get_speaker_encoder(cfg, device)

    sr = cfg.audio.sr
    mix = load_16k_mono(args.mix, sr)
    n = mix.numel() - (mix.numel() % cfg.audio.hop)
    mix = mix[:n]

    out = torch.zeros_like(mix)
    for path in args.enroll:
        ew = load_16k_mono(path, sr)
        emb = spk_enc.embed(ew.unsqueeze(0), torch.ones(1))
        if args.streaming:
            session = StreamingSession(model, stft, emb.squeeze(0), device)
            y = session.process(mix).cpu()
        else:
            with torch.no_grad():
                spec = stft.stft(mix.unsqueeze(0).to(device))
                est_spec, _ = model(spec, emb)
                y = stft.istft(est_spec, n).squeeze(0).cpu()
        out += y
        print(f"[infer] {path}: output rms {y.pow(2).mean().sqrt():.4f}")

    out = out.clamp(-1.0, 1.0)
    sf.write(args.out, out.numpy(), sr)
    print(f"[infer] wrote {args.out} ({n / sr:.1f}s, {len(args.enroll)} speaker(s))")


if __name__ == "__main__":
    main()
