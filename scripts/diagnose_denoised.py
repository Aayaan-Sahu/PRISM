"""
Diagnostic: Does denoising fix speaker similarity in noisy conditions?

    uv run python scripts/diagnose_denoised.py
"""
from __future__ import annotations
import os, sys, time, numpy as np, sounddevice as sd, torch, torchaudio
import noisereduce as nr

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]
from speechbrain.inference.speaker import EncoderClassifier

SR = 16_000
device = torch.device("cpu")

print("Loading ECAPA-TDNN …")
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(os.path.dirname(__file__), "..", "pretrained_models", "spkrec-ecapa-voxceleb"),
    run_opts={"device": str(device)},
)
print("✓ Ready\n")

def record(label, dur=4):
    input(f"  Press Enter → {label} ({dur}s) … ")
    print(f"  🎙 Recording …")
    a = sd.rec(int(dur * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    print("  ✓ Done.\n")
    return a[:, 0]

def denoise(audio):
    t0 = time.perf_counter()
    cleaned = nr.reduce_noise(y=audio, sr=SR, stationary=True, prop_decrease=0.9)
    ms = (time.perf_counter() - t0) * 1000
    return cleaned, ms

def embed(audio):
    w = torch.from_numpy(audio).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return model.encode_batch(w).squeeze()

def sim(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ── Enrollment (clean) ───────────────────────────────────────────────
print("═" * 55)
print("  ENROLLMENT: Speak clearly in a quiet environment")
print("═" * 55)

# Try to reuse target.npy
target_path = os.path.join(os.path.dirname(__file__), "..", "noise_gate", "target.npy")
if os.path.exists(target_path):
    print(f"  Using saved target.npy")
    clean_emb = torch.from_numpy(np.load(target_path)).float().to(device)
else:
    clean_audio = record("speak clearly (quiet room)")
    clean_emb = embed(clean_audio)

# ── Test: Noisy audio — RAW vs DENOISED ──────────────────────────────
print("═" * 55)
print("  TEST: Speak with BACKGROUND NOISE (music, people, etc)")
print("═" * 55)
noisy_audio = record("speak through noise")

# Raw embedding
raw_emb = embed(noisy_audio)
raw_sim = sim(clean_emb, raw_emb)

# Denoised embedding
denoised_audio, denoise_ms = denoise(noisy_audio)
denoised_emb = embed(denoised_audio)
denoised_sim = sim(clean_emb, denoised_emb)

print(f"  📊 RAW (no denoising):       sim = {raw_sim:.3f}")
print(f"  📊 DENOISED (noisereduce):   sim = {denoised_sim:.3f}")
print(f"  📊 Improvement:              +{denoised_sim - raw_sim:.3f}")
print(f"  ⏱  Denoise time:             {denoise_ms:.0f} ms for {len(noisy_audio)/SR:.0f}s audio\n")

# ── Test: Someone ELSE speaking ──────────────────────────────────────
print("═" * 55)
print("  TEST: Have your FRIEND speak (you stay quiet)")
print("═" * 55)
other_audio = record("friend speaks")

other_raw_emb = embed(other_audio)
other_raw_sim = sim(clean_emb, other_raw_emb)

other_denoised, _ = denoise(other_audio)
other_denoised_emb = embed(other_denoised)
other_denoised_sim = sim(clean_emb, other_denoised_emb)

print(f"  📊 FRIEND RAW:               sim = {other_raw_sim:.3f}")
print(f"  📊 FRIEND DENOISED:          sim = {other_denoised_sim:.3f}\n")

# ── Summary ──────────────────────────────────────────────────────────
print("═" * 55)
print("SUMMARY — Does denoising help?")
print("═" * 55)
print(f"  YOU (noisy, raw):      {raw_sim:.3f}")
print(f"  YOU (noisy, denoised): {denoised_sim:.3f}  {'✅ IMPROVED' if denoised_sim > raw_sim else '⚠️ NO HELP'}")
print(f"  FRIEND (raw):          {other_raw_sim:.3f}")
print(f"  FRIEND (denoised):     {other_denoised_sim:.3f}")
gap_raw = raw_sim - other_raw_sim
gap_denoised = denoised_sim - other_denoised_sim
print(f"\n  Gap (you vs friend):")
print(f"    Raw:      {gap_raw:+.3f}")
print(f"    Denoised: {gap_denoised:+.3f}  {'✅ BETTER SEPARATION' if gap_denoised > gap_raw else '⚠️ SIMILAR'}")
print(f"\n  Suggested threshold: {(denoised_sim + other_denoised_sim) / 2:.3f}")
