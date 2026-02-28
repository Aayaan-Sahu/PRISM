"""
Quick diagnostic: verify embedding quality in clean vs noisy conditions.

    uv run python scripts/diagnose_embeddings.py
"""

from __future__ import annotations
import os, sys, numpy as np, sounddevice as sd, torch, torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]
from speechbrain.inference.speaker import EncoderClassifier

SR = 16_000
device = torch.device("cpu")

print("Loading model …")
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "spkrec-ecapa-voxceleb"),
    run_opts={"device": str(device)},
)
print("✓ Ready\n")

def record(label, dur=4):
    input(f"  Press Enter to record {label} ({dur}s) … ")
    print(f"  🎙  Recording …")
    a = sd.rec(int(dur * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print("  ✓  Done.\n")
    return a[:, 0]

def embed(audio):
    w = torch.from_numpy(audio).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return model.encode_batch(w).squeeze()

def sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ── Test 1: Clean enrollment → clean test ────────────────────────────
print("═" * 50)
print("TEST 1: Clean → Clean  (same person, quiet room)")
print("═" * 50)
e1 = embed(record("ENROLLMENT (quiet, speak normally)"))
e2 = embed(record("TEST (quiet, speak again)"))
s = sim(e1, e2)
print(f"  ➜ Similarity: {s:.3f}  {'✅ GOOD' if s > 0.5 else '⚠️ LOW'}\n")

# ── Test 2: Clean enrollment → noisy test ────────────────────────────
print("═" * 50)
print("TEST 2: Clean → Noisy  (same person, ADD background noise)")
print("═" * 50)
print("  Play music, have someone else talk, etc.")
e3 = embed(record("TEST (noisy, speak through the noise)"))
s2 = sim(e1, e3)
print(f"  ➜ Similarity: {s2:.3f}  {'✅ OK' if s2 > 0.3 else '⚠️ LOW'}")
print(f"  ➜ Drop from clean: {s - s2:.3f}\n")

# ── Test 3: Clean enrollment → silence/noise only ───────────────────
print("═" * 50)
print("TEST 3: Clean → Noise ONLY  (DON'T speak, just background)")
print("═" * 50)
e4 = embed(record("NOISE ONLY (don't talk, just background noise)"))
s3 = sim(e1, e4)
print(f"  ➜ Similarity: {s3:.3f}  {'✅ GOOD' if s3 < 0.2 else '⚠️ HIGH'}")

# ── Also check saved target.npy ──────────────────────────────────────
target_path = os.path.join(os.path.dirname(__file__), "..", "noise_gate", "target.npy")
if os.path.exists(target_path):
    t = torch.from_numpy(np.load(target_path)).float().to(device)
    print(f"\n  vs target.npy: clean={sim(t, e2):.3f}, noisy={sim(t, e3):.3f}, noise_only={sim(t, e4):.3f}")

print(f"\n{'═' * 50}")
print(f"SUMMARY")
print(f"  Clean→Clean:     {s:.3f}")
print(f"  Clean→Noisy:     {s2:.3f}  (this is your effective range)")
print(f"  Clean→NoiseOnly: {s3:.3f}  (this is the floor)")
print(f"  Suggested threshold: {(s2 + s3) / 2:.3f}")
print(f"{'═' * 50}")
