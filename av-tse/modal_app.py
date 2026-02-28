"""
Modal App — Dolphin AV-TSE on Cloud GPU.

Deploys the Dolphin audio-visual speech separation model to a Modal A10G GPU.
The model weights and face detection weights are baked into the container image
so cold starts are fast and there's no re-downloading.

Usage:
    # Deploy (one-time, builds the image):
    modal deploy av-tse/modal_app.py

    # Run separation from your local machine:
    modal run av-tse/modal_app.py --input recordings/recording.mp4

    # Or call from Python (see separate.py for integration):
    from modal import Function
    fn = Function.from_name("dolphin-av-tse", "DolphinSeparator.separate")
    result = fn.remote(video_bytes)
"""

import modal
import os
import sys

# ─── Image Definition ────────────────────────────────────────────────────
# This builds a Docker-like container with all Dolphin dependencies.
# Modal caches this — it only rebuilds when dependencies change.

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
    # Mount the entire Dolphin repo into the container image
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "..", "Dolphin"),
        remote_path="/app/Dolphin",
    )
)

# ─── App ──────────────────────────────────────────────────────────────────

app = modal.App("dolphin-av-tse", image=dolphin_image)


# ─── GPU-Accelerated Separator ───────────────────────────────────────────

@app.cls(
    gpu="A10G",                      # Cheapest GPU that handles Dolphin well
    timeout=180,                     # 3 min max per call (safety net)
    scaledown_window=60,             # Release GPU after 60s of inactivity
)
class DolphinSeparator:
    """
    Loads Dolphin once on container start, then processes videos on demand.
    The GPU is automatically released after `container_idle_timeout` seconds 
    of no calls.
    """

    @modal.enter()
    def load_model(self):
        """Runs ONCE when the container cold-starts. Model stays in VRAM."""
        import torch

        # Force TensorFlow to use legacy Keras 2 (tf-keras) instead of Keras 3.
        # retina-face was written for Keras 2 and breaks with Keras 3's KerasTensor API.
        os.environ["TF_USE_LEGACY_KERAS"] = "1"

        # Force TensorFlow (used by RetinaFace face detector) to CPU only.
        # Modal's cuDNN 9.1.0 doesn't match TF's compiled cuDNN 9.3.0.
        # PyTorch uses its own CUDA/cuDNN and is unaffected.
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')

        # Make Dolphin importable
        sys.path.insert(0, "/app/Dolphin")

        from look2hear.models import Dolphin

        print("[Modal] Loading Dolphin model from HuggingFace...")
        self.model = Dolphin.from_pretrained("JusperLee/Dolphin")
        self.model.cuda().eval()
        print(f"[Modal] Model loaded on {next(self.model.parameters()).device}")

    @modal.method()
    def separate(self, video_bytes: bytes, num_speakers: int = 1) -> dict:
        """
        Receives a raw .mp4 as bytes, runs the full Dolphin pipeline,
        and returns a dict mapping speaker names to their isolated .wav bytes.

        Returns:
            {"speaker1": <wav_bytes>, "speaker2": <wav_bytes>, ...}
        """
        import tempfile
        import shutil

        sys.path.insert(0, "/app/Dolphin")
        from Inference import process_video

        # Write incoming video to a temp file
        work_dir = tempfile.mkdtemp(prefix="dolphin_")
        input_path = os.path.join(work_dir, "input.mp4")
        output_dir = os.path.join(work_dir, "output")

        with open(input_path, "wb") as f:
            f.write(video_bytes)

        print(f"[Modal] Processing {len(video_bytes)} bytes, {num_speakers} speaker(s)...")

        # Run Dolphin's full pipeline (face detect → lip crop → separation)
        process_video(
            input_file=input_path,
            output_path=output_dir,
            number_of_speakers=num_speakers,
            detect_every_N_frame=8,
            scalar_face_detection=1.5,
            cuda_device=0,
        )

        # Collect the separated .wav files
        results = {}
        for i in range(num_speakers):
            wav_path = os.path.join(output_dir, f"speaker{i+1}_est.wav")
            if os.path.exists(wav_path):
                with open(wav_path, "rb") as f:
                    results[f"speaker{i+1}"] = f.read()
                print(f"[Modal] speaker{i+1}_est.wav: {os.path.getsize(wav_path)} bytes")

        # Cleanup
        shutil.rmtree(work_dir, ignore_errors=True)

        return results


# ─── CLI entry point for quick testing ────────────────────────────────────

@app.local_entrypoint()
def main(speakers: int = 1, output: str = "recordings/recording.mp4"):
    """
    Full end-to-end flow:
      1. Opens the webcam recorder (press SPACE to start/stop)
      2. Uploads the recording to Modal
      3. Returns the separated audio

    Usage:
        modal run av-tse/modal_app.py
        modal run av-tse/modal_app.py --speakers 2
    """
    # --- Step 1: Record locally ---
    # Import here because record.py uses sounddevice/opencv which are local-only
    sys.path.insert(0, os.path.dirname(__file__))
    from record import Recorder

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Dolphin AV-TSE  (Record → Modal GPU → Result)  ║")
    print("╚══════════════════════════════════════════════════╝\n")

    recorder = Recorder(output_path=output)
    recording_path = recorder.run()

    if not recording_path or not os.path.isfile(recording_path):
        print("No recording saved. Exiting.")
        return

    # --- Step 2: Upload to Modal ---
    file_size = os.path.getsize(recording_path)
    print(f"\n[Upload] Sending {recording_path} ({file_size / 1024:.0f} KB) to Modal GPU...")

    with open(recording_path, "rb") as f:
        video_bytes = f.read()

    separator = DolphinSeparator()
    results = separator.separate.remote(video_bytes, num_speakers=speakers)

    # --- Step 3: Save results locally ---
    out_dir = os.path.splitext(recording_path)[0] + "_modal_output"
    os.makedirs(out_dir, exist_ok=True)

    for name, wav_bytes in results.items():
        out_path = os.path.join(out_dir, f"{name}_est.wav")
        with open(out_path, "wb") as f:
            f.write(wav_bytes)
        print(f"[Result] Saved: {out_path} ({len(wav_bytes)} bytes)")

    print(f"\n✓ Done! Separated audio in {out_dir}/")

