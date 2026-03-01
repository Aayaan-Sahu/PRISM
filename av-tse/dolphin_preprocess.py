"""
Dolphin Preprocessing — extract lip crops for the Dolphin audio-visual
speech separation model.

Produces:
  - video.mp4 — 88×88 grayscale lip crops at 25 FPS
  - audio.wav — 16 kHz mono

Input spec matches Dolphin's DP-LipCoder:
  - 88×88 grayscale frames
  - 25 FPS video
  - 16 kHz mono audio
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from recorder import _find_ffmpeg

import cv2
import numpy as np

from recorder import RecordedClip, _write_wav


# Dolphin input spec
_LIP_CROP_SIZE = 96    # initial crop before resize
_LIP_OUTPUT_SIZE = 88  # final output size
_OUTPUT_FPS = 25

# MediaPipe mouth_centre keypoint index
_MOUTH_CENTRE_IDX = 3


def extract_lip_region(
    frame: np.ndarray,
    detection,
    crop_size: int = _LIP_CROP_SIZE,
    output_size: int = _LIP_OUTPUT_SIZE,
) -> np.ndarray | None:
    """
    Extract and preprocess the lip region from a frame.

    Uses the mouth_centre keypoint (MediaPipe index 3) to crop a square
    region around the lips, resize to output_size×output_size, and
    convert to grayscale.

    Returns an (output_size, output_size) uint8 grayscale array, or None
    if the detection is missing.
    """
    if detection is None:
        return None

    h, w = frame.shape[:2]

    # Get mouth centre in pixel coords
    kp = detection.keypoints[_MOUTH_CENTRE_IDX]
    mx = int(kp.x * w)
    my = int(kp.y * h)

    # Crop a square region centred on the mouth
    half = crop_size // 2
    x1 = max(0, mx - half)
    y1 = max(0, my - half)
    x2 = min(w, mx + half)
    y2 = min(h, my + half)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Resize to output size
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    return gray


def preprocess_for_dolphin(clip: RecordedClip, output_dir: str) -> tuple[str, str]:
    """
    Process a RecordedClip into Dolphin-compatible format.

    Produces:
      - {output_dir}/video.mp4 — 88×88 grayscale lip crops at 25 FPS
      - {output_dir}/audio.wav — 16 kHz mono

    Returns (video_path, audio_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "video.mp4")
    audio_path = os.path.join(output_dir, "audio.wav")

    # Write audio
    _write_wav(audio_path, clip.audio, clip.audio_rate)

    # Extract lip crops from each frame
    lip_frames: list[np.ndarray] = []
    for frame, det in zip(clip.frames, clip.detections):
        lip = extract_lip_region(frame, det)
        if lip is not None:
            lip_frames.append(lip)
        else:
            # Use a black frame as placeholder if detection is missing
            lip_frames.append(np.zeros((_LIP_OUTPUT_SIZE, _LIP_OUTPUT_SIZE), dtype=np.uint8))

    if not lip_frames:
        return video_path, audio_path

    # Write lip video — try ffmpeg, fall back to direct OpenCV
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(
                tmp_path, fourcc, _OUTPUT_FPS,
                (_LIP_OUTPUT_SIZE, _LIP_OUTPUT_SIZE),
                isColor=False,
            )
            for lip in lip_frames:
                writer.write(lip)
            writer.release()

            cmd = [
                ffmpeg, "-y",
                "-i", tmp_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                video_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Direct write — grayscale frames need BGR conversion for mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            video_path, fourcc, _OUTPUT_FPS,
            (_LIP_OUTPUT_SIZE, _LIP_OUTPUT_SIZE),
            isColor=True,
        )
        for lip in lip_frames:
            bgr = cv2.cvtColor(lip, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
        writer.release()

    return video_path, audio_path

