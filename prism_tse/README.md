# PRISM-TSE

Training scaffold for the PRISM streaming target-speaker-extraction model: a
causal, speaker-conditioned complex-masking network (~2M params) that replaces
the WeSep + per-chunk ECAPA gate pipeline. Enroll a speaker once (frozen
ECAPA-TDNN embedding — same encoder the main repo already uses), then stream
16 ms hops through the model; multi-speaker = one inference pass per enrolled
embedding, outputs summed. Algorithmic latency: 32 ms.

Plain PyTorch, single GPU (H200). No Modal.

## Setup (H200 server)

```bash
git clone <repo> && cd <repo>          # branch: prism-tse
python -m venv .venv && source .venv/bin/activate
pip install -r prism_tse/requirements.txt

# Pre-download the frozen ECAPA encoder ON A NODE WITH INTERNET (login node):
python -m prism_tse.models.ecapa --predownload \
    --savedir data_root/pretrained/spkrec-ecapa-voxceleb
# Then in job scripts (compute nodes often have no internet):
export HF_HUB_OFFLINE=1
```

## Data (starter recipe, ~25 GB)

```bash
python -m prism_tse.data.download librispeech  --data_root data_root
python -m prism_tse.data.download musan        --data_root data_root
python -m prism_tse.data.download rirs_openslr --data_root data_root

python -m prism_tse.data.build_manifests --data_root data_root
python -m prism_tse.data.generate_rirs   --data_root data_root --n-rooms 500
```

Scaling up later: `download librispeech --also-360` (+23 GB, ~1,150 speakers),
`download dns` (best-effort URLs), `download vctk` — then rerun
`build_manifests`. Training data is synthesized on the fly (speech + speakers
talking over each other + noise + reverb); nothing is pre-rendered.

Sanity-listen to what the mixer produces:

```bash
python -m prism_tse.datasets.mixer --config prism_tse/configs/default.yaml \
    --data_root data_root --out mixer_examples --n 10
```

## Train

```bash
# quick pipeline check (a few hundred steps, watch loss decrease):
python -m prism_tse.train --config prism_tse/configs/default.yaml \
    --data_root data_root --run_dir runs/v1

# resume after preemption:
python -m prism_tse.train --config prism_tse/configs/default.yaml \
    --data_root data_root --run_dir runs/v1 --resume auto

# monitor (from your laptop):
ssh -L 6006:localhost:6006 <server>  &&  tensorboard --logdir runs/v1/tb
```

Watch three validation numbers:
- `val/si_sdri` — separation quality on target-present mixtures (>5 dB after
  ~50k steps is on track),
- `val/leakage_db` — output energy when only non-enrolled speakers talk
  (should sink below −25 dB: that's the noise-gate behavior),
- `val/word_pres` — fraction of target-active frames preserved within 10 dB
  (the anti-word-clipping metric).

## Listening tests

```bash
python -m prism_tse.infer_wav --ckpt runs/v1/ckpt/best.pt \
    --mix demo-recordings/raw_audio_XXXX.wav \
    --enroll my_voice.wav --out filtered.wav
# multi-speaker: repeat --enroll; add --streaming to use the hop-by-hop path
```

## Tests

```bash
pytest prism_tse/tests/ -q
```

`test_streaming_parity.py` is the load-bearing one: it asserts the offline
training forward and the stateful hop-by-hop streaming path produce identical
output (atol 1e-4). If you change the model or STFT, this must stay green.

## Layout

```
config.py / configs/       config dataclasses + YAML recipes
data/                      download, manifest building, RIR bank generation
datasets/                  on-the-fly mixture synthesis (mixer.py) + augments
models/stft.py             causal STFT (owned framing — NOT torch.stft)
models/separator.py        PrismTSE: FiLM-conditioned causal LSTM masker
models/ecapa.py            frozen ECAPA wrapper (+ offline predownload)
losses.py                  SI-SDR + asymmetric spectral + silence losses
train.py / validate.py     training loop, metrics
streaming.py               StreamingSession (deployment inference path)
infer_wav.py               checkpoint -> filtered wav CLI
```
