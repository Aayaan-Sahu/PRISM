import os
import sys
import argparse
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOLPHIN_DIR = os.path.join(_REPO_ROOT, "Dolphin")
sys.path.insert(0, _DOLPHIN_DIR)

from Inference import process_video, resolve_device

try:
    from Inference_with_status import process_video_with_status
except Exception:
    process_video_with_status = None

def _status_callback(message):
    status = message.get("status", "")
    progress = message.get("progress")
    if progress is None:
        print(f"[Status] {status}")
    else:
        print(f"[Status {float(progress) * 100:5.1f}%] {status}")

def main():
    parser = argparse.ArgumentParser(description="Test Dolphin's ability to extract a voice from a noisy video.")
    parser.add_argument("--video", required=True, help="Path to .mp4 of a person talking with background noise")
    parser.add_argument("--output_dir", default="dolphin_test_results", help="Where to save the extracted voice")
    parser.add_argument("--speakers", type=int, default=1, help="Number of speakers to separate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                        help="Compute device to use (default: auto)")
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="CUDA device index when using CUDA (default: None)")
    parser.add_argument("--no-status", dest="use_status", action="store_false",
                        help="Use Inference.process_video instead of Inference_with_status.process_video_with_status")
    parser.set_defaults(use_status=True)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        requested_device = args.device
        if args.cuda_device is None and requested_device == "cuda":
            args.cuda_device = 0
        selected_device = resolve_device(requested_device, args.cuda_device)

        print(f"[Test] Running Dolphin on {args.video} ...")
        print(f"[Test] Requested device: {requested_device}")
        print(f"[Test] Selected device: {selected_device}")
        print(f"[Test] Speakers: {args.speakers}")
        
        if args.use_status and process_video_with_status is not None:
            print("[Test] Using process_video_with_status")
            output_files = process_video_with_status(
                input_file=os.path.abspath(args.video),
                output_path=output_dir,
                number_of_speakers=args.speakers,
                detect_every_N_frame=8,
                scalar_face_detection=1.5,
                cuda_device=args.cuda_device,
                device=requested_device,
                status_callback=_status_callback,
            )
        else:
            if args.use_status and process_video_with_status is None:
                print("[Test] Inference_with_status is unavailable; falling back to process_video")
            print("[Test] Using process_video")
            output_files = process_video(
                input_file=os.path.abspath(args.video),
                output_path=output_dir,
                number_of_speakers=args.speakers,
                detect_every_N_frame=8,
                scalar_face_detection=1.5,
                cuda_device=args.cuda_device,
                device=requested_device,
            )
        
        print("\n" + "="*50)
        print("✓ Test Complete!")
        print("Dolphin generated the following outputs:")
        missing_outputs = []
        for i in range(args.speakers):
            wav_path = os.path.join(output_dir, f"speaker{i+1}_est.wav")
            mp4_path = os.path.join(output_dir, f"s{i+1}.mp4")
            if os.path.exists(wav_path):
                print(f"  - Audio: {wav_path}")
            else:
                missing_outputs.append(wav_path)
                print(f"  - Missing audio: {wav_path}")
            if os.path.exists(mp4_path):
                print(f"  - Video: {mp4_path}")
            else:
                missing_outputs.append(mp4_path)
                print(f"  - Missing video: {mp4_path}")

        if output_files:
            print("[Test] Returned video list:")
            for p in output_files:
                print(f"  - {p}")

        if missing_outputs:
            raise RuntimeError(f"Missing expected outputs ({len(missing_outputs)} files).")
        print("="*50)

    except Exception:
        traceback.print_exc()
        e = sys.exc_info()[1]
        print(f"Error during Dolphin test: {e}")

main()
