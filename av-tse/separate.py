"""
AV-TSE Separator — runs Dolphin on a recorded .mp4 to extract the target speaker.

Takes the .mp4 output from record.py and passes it through Dolphin's
pre-trained audio-visual speech separation model.  Outputs isolated
speaker audio as .wav files.

Usage:
    python separate.py recordings/recording.mp4
    python separate.py recordings/recording.mp4 --speakers 2
"""

from __future__ import annotations

import argparse
import os
import sys

# Add the Dolphin repo to the Python path so we can import its modules
DOLPHIN_DIR = os.path.join(os.path.dirname(__file__), "..", "Dolphin")
sys.path.insert(0, os.path.abspath(DOLPHIN_DIR))

from Inference import process_video
from ecapa_enroll import enroll_from_wav


def main():
    parser = argparse.ArgumentParser(
        description="Run Dolphin AV-TSE on a recorded video to extract speaker voice(s)."
    )
    parser.add_argument("input", help="Path to the input .mp4 file (from record.py)")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: <input_name>_separated/)"
    )
    parser.add_argument(
        "-s", "--speakers", type=int, default=1,
        help="Number of speakers to separate (default: 1)"
    )
    parser.add_argument(
        "--detect-every-n", type=int, default=8,
        help="Run face detection every N frames (default: 8)"
    )
    parser.add_argument(
        "--face-scale", type=float, default=1.5,
        help="Face bounding box scale factor (default: 1.5)"
    )
    parser.add_argument(
        "--enroll", action="store_true",
        help="After separation, run ECAPA-TDNN on speaker WAVs and save embeddings to noise_gate/embeddings/"
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    # Default output directory
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input) or ".", f"{base}_separated")

    print("=" * 60)
    print("  Dolphin AV-TSE Separator")
    print("=" * 60)
    print(f"  Input:    {args.input}")
    print(f"  Output:   {args.output}")
    print(f"  Speakers: {args.speakers}")
    print("=" * 60)
    print()

    # Run Dolphin's full pipeline:
    #   1. Convert video to 25 FPS
    #   2. Detect & track faces via retina-face + face_alignment
    #   3. Crop mouth ROIs
    #   4. Load Dolphin model from HuggingFace
    #   5. Run audio-visual separation
    #   6. Save isolated speaker audio + tracked video
    output_files = process_video(
        input_file=args.input,
        output_path=args.output,
        number_of_speakers=args.speakers,
        detect_every_N_frame=args.detect_every_n,
        scalar_face_detection=args.face_scale,
        cuda_device=None,   # our patched Inference.py auto-detects device
    )

    print("\n" + "=" * 60)
    print("  Separation complete!")
    print("=" * 60)
    for i, f in enumerate(output_files):
        wav_path = os.path.join(args.output, f"speaker{i+1}_est.wav")
        print(f"  Speaker {i+1} audio: {wav_path}")
        print(f"  Speaker {i+1} video: {f}")
    print("=" * 60)

    # ── Flow 2 integration: enroll speaker WAVs via ECAPA-TDNN ─────────
    if args.enroll:
        print("\n" + "=" * 60)
        print("  ECAPA-TDNN Enrollment")
        print("=" * 60)
        enrolled_any = False
        for i in range(args.speakers):
            wav_path = os.path.join(args.output, f"speaker{i+1}_est.wav")
            if os.path.isfile(wav_path):
                print(f"  Enrolling speaker {i+1} from {wav_path} …")
                try:
                    emb_path = enroll_from_wav(wav_path)
                    print(f"  ✓ Embedding saved → {emb_path}")
                    enrolled_any = True
                except Exception as exc:
                    print(f"  ⚠  Enrollment failed for speaker {i+1}: {exc}")
            else:
                print(f"  ⚠  WAV not found for speaker {i+1}: {wav_path}")
        if not enrolled_any:
            print("  No embeddings were saved.")
        print("=" * 60)


if __name__ == "__main__":
    main()
