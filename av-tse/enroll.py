"""
enroll.py — One-command speaker enrollment pipeline
====================================================

Runs Flow 1 + Flow 2 end-to-end:

  1. Opens the webcam (press 'i' to start / stop recording)
  2. Saves the face-crop video + mic audio as recordings/recording.mp4
  3. Passes the .mp4 through Dolphin to isolate the speaker's voice
  4. Runs ECAPA-TDNN on the isolated audio
  5. Saves the 192-dim embedding to noise_gate/embeddings/embeddingN.npy

Usage:
    cd /path/to/spatial-audio-recognition
    PYTHONPATH=av-tse:. uv run python av-tse/enroll.py

Optional flags:
    --output   recordings/recording.mp4   custom recording path
    --camera   0                          camera index
    --speakers 1                          number of speakers Dolphin should find
"""

from __future__ import annotations

import argparse
import os
import sys

# ── Make av-tse/ importable (record, separate, ecapa_enroll, etc.) ──────────
_AVTSE_DIR = os.path.dirname(__file__)
_REPO_ROOT  = os.path.abspath(os.path.join(_AVTSE_DIR, ".."))
for _p in (_AVTSE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Make Dolphin/ importable ─────────────────────────────────────────────────
_DOLPHIN_DIR = os.path.join(_REPO_ROOT, "Dolphin")
if os.path.isdir(_DOLPHIN_DIR) and _DOLPHIN_DIR not in sys.path:
    sys.path.insert(0, _DOLPHIN_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full enrollment pipeline: record → Dolphin → ECAPA embedding"
    )
    parser.add_argument(
        "--output", default=os.path.join(_REPO_ROOT, "recordings", "recording.mp4"),
        help="Path for the recorded .mp4 (default: recordings/recording.mp4)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--speakers", type=int, default=1,
        help="Number of speakers Dolphin should separate (default: 1)"
    )
    args = parser.parse_args()

    # ── Step 1: Record ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 / 3 — Recording")
    print("  Press 'i' to START recording.")
    print("  It will automatically record for exactly 5 seconds.")
    print("  Press Q or Esc to quit without saving.")
    print("=" * 60 + "\n")

    from record import Recorder
    recorder = Recorder(output_path=args.output, camera_index=args.camera)
    recording_path = recorder.run()

    if not recording_path or not os.path.isfile(recording_path):
        print("\n❌  No recording saved. Exiting.")
        sys.exit(1)

    print(f"\n✓  Recording saved: {recording_path}")

    # ── Step 2: Dolphin separation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2 / 3 — Dolphin speech separation")
    print("=" * 60 + "\n")

    base = os.path.splitext(os.path.basename(recording_path))[0]
    dolphin_out = os.path.join(os.path.dirname(recording_path), f"{base}_separated")

    try:
        from Inference import process_video
        print(f"[Dolphin] Processing {recording_path} → {dolphin_out} …")
        process_video(
            input_file=recording_path,
            output_path=dolphin_out,
            number_of_speakers=args.speakers,
            detect_every_N_frame=8,
            scalar_face_detection=1.5,
            cuda_device=None,   # auto-detect
        )
        print(f"[Dolphin] ✓ Separation complete → {dolphin_out}")
    except ImportError:
        print("⚠️  Dolphin/Inference.py not found — skipping Dolphin separation.")
        print("   The Dolphin/ directory appears to be empty.")
        print("   Clone the Dolphin repo first: https://github.com/JusperLee/Dolphin")
        print("   For now, attempting to enroll directly from the raw recording audio…\n")
        # Fall back: extract audio from the mp4 and enroll directly
        dolphin_out = None

    # ── Step 3: ECAPA-TDNN enrollment ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3 / 3 — ECAPA-TDNN speaker embedding")
    print("=" * 60 + "\n")

    from ecapa_enroll import enroll_from_wav
    import glob

    wav_to_enroll: str | None = None

    if dolphin_out and os.path.isdir(dolphin_out):
        # Prefer Dolphin-separated speaker WAV
        candidates = sorted(glob.glob(os.path.join(dolphin_out, "speaker*_est.wav")))
        if candidates:
            wav_to_enroll = candidates[0]
            print(f"[Enroll] Using Dolphin-separated: {os.path.basename(wav_to_enroll)}")

    if wav_to_enroll is None:
        # Fallback: extract the audio track from the mp4
        import tempfile, subprocess
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        print(f"[Enroll] Extracting audio from {os.path.basename(recording_path)} …")
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", recording_path,
             "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", tmp_wav],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if ret.returncode == 0:
            wav_to_enroll = tmp_wav
            print(f"[Enroll] Audio extracted to temp file.")
        else:
            print("❌  ffmpeg failed to extract audio. Exiting.")
            sys.exit(1)

    emb_path = enroll_from_wav(wav_to_enroll)

    print("\n" + "=" * 60)
    print("  ✅  Enrollment complete!")
    print(f"  Embedding saved → {emb_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
