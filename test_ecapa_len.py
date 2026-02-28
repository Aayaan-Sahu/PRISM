import torch
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]
import time
from speechbrain.inference.speaker import EncoderClassifier

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/Users/donde783985/Desktop/hackathon/spatial-audio-recognition/pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": str(device)}
)

for duration in [0.3, 0.15, 0.1, 0.08, 0.05]:
    length = int(16000 * duration)
    wav = torch.randn(1, length).to(device)
    # Warmup
    for _ in range(5):
        _ = model.encode_batch(wav)
    
    # Measure
    t0 = time.time()
    for _ in range(50):
        _ = model.encode_batch(wav)
    t1 = time.time()
    avg_ms = ((t1-t0)/50)*1000
    print(f"Duration {duration}s: Success - avg inference {avg_ms:.1f}ms")
