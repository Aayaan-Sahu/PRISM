"""
Target Speaker Extraction (TSE) Simulation Script
=================================================

Since native Target Speaker Extraction (TSE) models like SpEx+ or 
VoiceFilter require incredibly heavy external dependencies (ModelScope, 
older Transformers, etc.), this script simulates the EXACT logic of TSE 
locally on your Mac using the models you already have installed!

It simulates your proposed logic:
  "each sepformer track runs a similarity with the embeddings... 
   the final audio output should just be the input signal from 1... 
   we don't even need to consider 3 or any other background information."

Usage
-----
    uv run python noise_gate/test_tse.py --target noise_gate/recordings/Tanay.wav --mixture noise_gate/recordings/mixture.wav
"""

import argparse
import os
import sys
import time

import torch
import torchaudio
import torch.nn.functional as F

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.speaker import EncoderClassifier

# =====================================================================
# Configuration
# =====================================================================

MODEL_SR = 16000
SIMILARITY_THRESHOLD = 0.25 # standard threshold for ECAPA-TDNN

# Directories
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "..", "pretrained_models")
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_tse")

def main():
    parser = argparse.ArgumentParser(description="Test Target Speaker Extraction Logic")
    parser.add_argument(
        "--target",
        type=str,
        default=os.path.join(RECORDINGS_DIR, "Tanay.wav"),
        help="Path to the clean enrollment audio of the Target Speaker (e.g., Tanay.wav)",
    )
    parser.add_argument(
        "--mixture",
        type=str,
        default=os.path.join(RECORDINGS_DIR, "mixture.wav"),
        help="Path to the noisy audio containing the Target + Background speakers",
    )
    args = parser.parse_args()

    # 1. Validate inputs
    if not os.path.exists(args.target):
        print(f"❌  Target enrollment file not found: {args.target}")
        print(f"   Please record a 5-10 second clear clip of your voice and save it here.")
        sys.exit(1)

    if not os.path.exists(args.mixture):
        print(f"❌  Mixture (noisy) file not found: {args.mixture}")
        print(f"   Please place a test file with overlapping voices here.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Using device: {device}\n")

    # 2. Load Models
    print("⏳ Loading SepFormer Separation Model ...", flush=True)
    sep_model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr16k",
        savedir=os.path.join(PRETRAINED_DIR, "sepformer-whamr16k"),
        run_opts={"device": device},
    )

    print("⏳ Loading ECAPA-TDNN Speaker Recognition Model ...", flush=True)
    ecapa_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(PRETRAINED_DIR, "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
    )
    print("✅ Models loaded successfully.\n")

    # 3. Load Audio
    print(f"🎧 Loading Target Reference: {os.path.basename(args.target)}")
    target_waveform, sr_target = torchaudio.load(args.target)
    
    print(f"🎧 Loading Noisy Mixture:  {os.path.basename(args.mixture)}")
    mix_waveform, sr_mix = torchaudio.load(args.mixture)

    # Convert to mono
    if target_waveform.shape[0] > 1:
        target_waveform = target_waveform.mean(dim=0, keepdim=True)
    if mix_waveform.shape[0] > 1:
        mix_waveform = mix_waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr_target != MODEL_SR:
        target_waveform = torchaudio.transforms.Resample(sr_target, MODEL_SR)(target_waveform)
    if sr_mix != MODEL_SR:
        mix_waveform = torchaudio.transforms.Resample(sr_mix, MODEL_SR)(mix_waveform)

    target_waveform = target_waveform.to(device)
    mix_waveform = mix_waveform.to(device)

    # 4. Process Target Embedding (The Enrollment Phase)
    with torch.no_grad():
        target_emb = ecapa_model.encode_batch(target_waveform)
        target_emb = target_emb.squeeze(1) # shape: [1, 192]

    # 5. Extract (The TSE Phase)
    print(f"\n🚀 Running Extraction Pipeline (Mixture length: {mix_waveform.shape[1] / MODEL_SR:.1f}s) ...")
    t0 = time.perf_counter()

    with torch.no_grad():
        # Step A: Separate ALL voices blindly (returns 2 tracks)
        est_sources = sep_model.separate_batch(mix_waveform)
        src1 = est_sources[0, :, 0] # shape: [T]
        src2 = est_sources[0, :, 1] # shape: [T]

        # Step B: Identify the voices using ECAPA
        sources_for_ecapa = torch.stack([src1, src2], dim=0)
        src_embeddings = ecapa_model.encode_batch(sources_for_ecapa).squeeze(1) # [2, 192]

        sims = F.cosine_similarity(src_embeddings.unsqueeze(1), target_emb.unsqueeze(0), dim=-1)
        max_sims, _ = sims.max(dim=1) # [2]

        print(f"   [Track 1] vs Target Similarity: {max_sims[0].item():.2f}")
        print(f"   [Track 2] vs Target Similarity: {max_sims[1].item():.2f}")

        # Step C: Combine only the tracks that match the target! 
        # (This is the TSE logic)
        final_output_tracks = []
        if max_sims[0].item() > SIMILARITY_THRESHOLD:
            final_output_tracks.append(src1)
            print("   ✅ Track 1 MATCHES the Target! Selecting Track 1.")
        
        if max_sims[1].item() > SIMILARITY_THRESHOLD:
            final_output_tracks.append(src2)
            print("   ✅ Track 2 MATCHES the Target! Selecting Track 2.")
        
        if len(final_output_tracks) == 0:
            print("   ❌ NO MATCHES. The Target speaker is not present in the mixture. Outputting silence.")
            final_audio = torch.zeros_like(src1)
        else:
            final_audio = sum(final_output_tracks)

    t1 = time.perf_counter()
    print(f"⏱️  Pipeline finished in {t1 - t0:.2f} seconds.")

    # 6. Save output
    out_path = os.path.join(OUTPUT_DIR, f"filtered.wav")
    
    # Normalize volume before saving to prevent clipping
    final_audio = final_audio.cpu().unsqueeze(0) # [1, T]
    max_val = torch.max(torch.abs(final_audio))
    if max_val > 0:
        final_audio = (final_audio / max_val) * 0.9

    torchaudio.save(out_path, final_audio, MODEL_SR)
    
    # Save the original mixture right next to it so it's easy to compare
    torchaudio.save(os.path.join(OUTPUT_DIR, "original.wav"), mix_waveform.cpu(), MODEL_SR)

    print(f"\n💾 Saved extracted voice to:        {out_path}")
    print(f"💾 Saved original noisy mixture to: {os.path.join(OUTPUT_DIR, 'original.wav')}")


if __name__ == "__main__":
    main()
