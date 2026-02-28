"""
Shared configuration for the Selective Speaker Amplification system.
"""
from pathlib import Path

# ─── Hardware ────────────────────────────────────────────────────
AGG_DEVICE_INDEX = 9       # macOS Aggregate Device (4 in, 0 out)
CHANNELS = 4               # 2 ch per Yeti × 2 Yetis
SAMPLE_RATE = 48_000       # native rate of the aggregate device
TARGET_SR = 16_000         # model-expected sample rate (ECAPA, VAD)
SEPARATOR_SR = 8_000       # SepFormer expected sample rate

# ─── Microphone geometry ─────────────────────────────────────────
MIC_DISTANCE_M = 0.4953    # inter-mic spacing in metres
SPEED_OF_SOUND = 343.0     # m/s

# ─── Enrollment ──────────────────────────────────────────────────
ENROLL_DURATION_SEC = 7    # how long to record for enrollment
VOICEPRINT_DIR = Path("voiceprints")

# ─── Live pipeline ───────────────────────────────────────────────
CHUNK_SEC = 0.5            # processing chunk length (seconds)
OVERLAP_SEC = 0.25         # overlap between chunks for crossfading
G_FG = 1.0                 # foreground gain (enrolled speakers)
G_BG = 0.15                # background gain (unknown speakers)
ID_THRESHOLD = 0.25        # cosine-sim threshold for speaker ID
SMOOTHING_TAU = 0.05       # exponential gain smoothing time constant

# ─── Calibration / detection ─────────────────────────────────────
CALIBRATION_SEC = 2.0
SPIKE_SIGMA = 3.0
MIN_CONFIDENCE = 1.5
WINDOW_SEC = 0.20

EPS = 1e-12
