import os
import sys
import modal

# Docker like contianer with all dolphin dependencies
# This is cached by modal - only rebuilds when deps change
dolphin_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")  # OpenCV + ffmpeg deps
    .pip_install(
        "torch==2.5.0",
        "torchvision==0.20.0",
        "torchaudio==2.5.0",
        "moviepy==2.1.2",
        "face_alignment",
        "beartype",
        "taylor_series_linear_attention",
        "av",
        "huggingface_hub==0.36.2",
        "einops",
        "vector_quantize_pytorch",
        "scikit-image",
        "retina-face",
        "safetensors",
        "tf-keras",
        "opencv-python-headless",
        "numpy<2",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "Dolphin"),
        remote_path="/app/Dolphin",
    )
)

app = modal.App("dolphin-inference", image=dolphin_image)

@app.cls(
    gpu="A100",
    timeout=180,            # 3 min max per call
    scaledown_window=60,    # release GPU after 60s of inactivity
)
class DolphinSeparator:
    @modal.enter()
    def load_model(self):
        # This function runs once when the container starts
        import torch

        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        sys.path.insert(0, "/app/Dolphin")

    @modal.method()
    def separate(self, video_bytes: bytes, num_speakers: int = 1, tracking_json_bytes: bytes = None) -> dict:
        import uuid
        import shutil
        import json
        from Inference_with_status import process_video_with_status

        # Create a temporary working directory
        work_dir = f"/tmp/dolphin_{uuid.uuid4().hex}"
        os.makedirs(work_dir, exist_ok=True)

        input_video_path = os.path.join(work_dir, "input.mp4")
        with open(input_video_path, "wb") as f:
            f.write(video_bytes)
            
        target_boxes = None
        if tracking_json_bytes is not None:
            target_boxes = json.loads(tracking_json_bytes)
            print(f"[Modal] Received targeted tracking JSON with {len(target_boxes)} frames.")
        
        output_dir = os.path.join(work_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # define a logger for dolphin inference
        def status_logger(message):
            status = message.get("status", "")
            progress = message.get("progress")
            if progress is None:
                print(f"[Modal Status] {status}")
            else:
                print(f"[Modal Status {float(progress) * 100:5.1f}%] {status}")

        print(f"[Modal] Running Dolphin inference on {num_speakers} speaker(s)...")
        # Run inference using CUDA
        output_files = process_video_with_status(
            input_file=input_video_path,
            output_path=output_dir,
            number_of_speakers=num_speakers,
            detect_every_N_frame=8,
            scalar_face_detection=1.5,
            cuda_device=0,
            device="cuda",
            status_callback=status_logger,
            target_boxes=target_boxes,
        )
        
        print("[Modal] Collecting output files...")
        results = {}
        for file_path in output_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    results[filename] = f.read()

        # Grab the isolated audio as well
        audio_paths = [os.path.join(output_dir, f"speaker{i+1}_est.wav") for i in range(num_speakers)]
        for audio_path in audio_paths:
            if os.path.exists(audio_path):
                filename = os.path.basename(audio_path)
                with open(audio_path, "rb") as f:
                    results[filename] = f.read()

        print(f"[Modal] Returning {len(results)} files.")
        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)
        return results

def main(video_path: str = "output_faces.mp4", num_speakers: int = 1, output_dir: str = "modal_output", tracking_path: str = "target_tracking.json"):
    if not os.path.exists(video_path):
        print(f"Error: Could not find input video '{video_path}'")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    job_output_dir = os.path.join(output_dir, base_name)

    print(f"Reading {video_path}...")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        
    tracking_json_bytes = None
    if os.path.exists(tracking_path):
        print(f"Reading target tracking data {tracking_path}...")
        with open(tracking_path, "rb") as f:
            tracking_json_bytes = f.read()

    print(f"Sending video to Modal for inference (Job: {base_name})...")
    with app.run():
        separator = DolphinSeparator()
        results = separator.separate.remote(video_bytes, num_speakers, tracking_json_bytes)

    if results:
        os.makedirs(job_output_dir, exist_ok=True)
        print(f"Writing outputs to {job_output_dir}/ ...")
        for filename, data in results.items():
            out_path = os.path.join(job_output_dir, filename)
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  Saved {out_path}")
        print("Done!")
    else:
        print("No results returned from Modal.")