import torch
import torchaudio
import numpy as np
import os

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from speechbrain.inference.speaker import EncoderClassifier

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}
    )
    
    target_emb = torch.from_numpy(np.load("noise_gate/target.npy")).float().to(device)
    
    # Load original audio
    wav, sr = torchaudio.load("output_original.wav")
    
    duration = 0.30
    chunk_len = int(sr * duration)
    best_300 = 0.0
    for start in range(0, wav.shape[1] - chunk_len, chunk_len//2):
        chunk = wav[:, start:start+chunk_len].to(device)
        with torch.no_grad():
            emb = model.encode_batch(chunk).squeeze()
        sim_t = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        if sim_t > best_300:
            best_300 = sim_t
            
    duration = 0.09
    chunk_len = int(sr * duration)
    best_090 = 0.0
    for start in range(0, wav.shape[1] - chunk_len, chunk_len//2):
        chunk = wav[:, start:start+chunk_len].to(device)
        with torch.no_grad():
            emb = model.encode_batch(chunk).squeeze()
        sim_t = torch.nn.functional.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0)).item()
        if sim_t > best_090:
            best_090 = sim_t
            
    print(f"Max similarity with 300ms buffer: {best_300:.3f}")
    print(f"Max similarity with  90ms buffer: {best_090:.3f}")

if __name__ == "__main__":
    main()
