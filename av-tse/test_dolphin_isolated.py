import os
import sys
import argparse
import tempfile

# Ensure Dolphin is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOLPHIN_DIR = os.path.join(_REPO_ROOT, "Dolphin")
sys.path.insert(0, _DOLPHIN_DIR)

from Inference import process_video


def main():
    parser = argparse.ArgumentParser(description="Test Dolphin's ability to extract a voice from a noisy video.")
    parser.add_argument("--video", required=True, help="Path to .mp4 of a person talking with background noise")
    parser.add_argument("--output_dir", default="dolphin_test_results", help="Where to save the extracted voice")
    parser.add_argument("--speakers", type=int, default=1, help="Number of speakers to separate")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print(f"[Test] Running Dolphin on {args.video} ...")
        
        # Run Dolphin
        output_files = process_video(
            input_file=os.path.abspath(args.video),
            output_path=os.path.abspath(args.output_dir),
            number_of_speakers=args.speakers,
            detect_every_N_frame=8,
            scalar_face_detection=1.5,
            cuda_device=None,
            skip_video_rendering=True  # Skip rendering the tracked video, we just want the audio
        )
        
        print("\n" + "="*50)
        print("✓ Test Complete!")
        print("Dolphin generated the following isolated audio:")
        for i in range(args.speakers):
            wav_path = os.path.join(args.output_dir, f"speaker{i+1}_est.wav")
            if os.path.exists(wav_path):
                print(f"  - {wav_path}")
            else:
                print(f"  - Error: {wav_path} not found")
        print("="*50)

    except Exception as e:
        print(f"Error during Dolphin test: {e}")

if __name__ == "__main__":
    main()
