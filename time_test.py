import torch
import time
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

from speechbrain.inference.speaker import EncoderClassifier

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/Users/donde783985/Desktop/hackathon/spatial-audio-recognition/pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": str(device)}
)

for recent in [0.08, 0.15, 0.25]:
    c_full = torch.randn(1, int(16000 * 0.34)).to(device)
    c_recent = torch.randn(1, int(16000 * recent)).to(device)
    
    pad = c_full.shape[1] - c_recent.shape[1]
    c_rec_pad = torch.nn.functional.pad(c_recent, (pad, 0))
    batch = torch.cat([c_full, c_rec_pad], dim=0)

    # warmup
    for _ in range(5):
        model.encode_batch(batch)

    t0 = time.perf_counter()
    for _ in range(50):
        model.encode_batch(batch)
    t1 = time.perf_counter()
    print(f"recent={recent}s -> {(t1-t0)/50*1000:.2f} ms")
