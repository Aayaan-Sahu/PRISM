"""
Targeting System (AV-TSE) — Extracts target angle and cropped face for neural speech extraction.

Upgraded to use MediaPipe Tasks API (FaceDetector) to track the target.
"""

from __future__ import annotations

import math
import os
import threading
import urllib.request
from contextlib import contextmanager
from typing import Generator, Iterator, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_PATH = "blaze_face_short_range.tflite"
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

def _ensure_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        print(f"Downloading face detector model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")

# ── constants ──────────────────────────────────────────────────────────────
HFOV_DEG: float = 60.0          # horizontal field-of-view of webcam
CENTER_X: float = 0.5           # normalised horizontal centre
CROSSHAIR_HALF: int = 20        # pixels for static crosshair arms

COLOR_TARGET   = (0,   255,   0)   # bright green
COLOR_BG_FACE  = (160, 160, 160)   # grey
COLOR_CROSS    = (255, 255, 255)   # white

# MediaPipe FaceDetector nose tip index
_NOSE_TIP_IDX = 2

# Neural network expected crop size
CROP_SIZE = (512, 512) 


def _angle_from_nose_x(nose_x: float) -> float:
    """Map normalised nose x-coordinate → azimuth angle in degrees."""
    return (nose_x - CENTER_X) * HFOV_DEG


def _draw_crosshair(frame, cx: int, cy: int) -> None:
    h = CROSSHAIR_HALF
    cv2.line(frame, (cx - h, cy), (cx + h, cy), COLOR_CROSS, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - h), (cx, cy + h), COLOR_CROSS, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, COLOR_CROSS, -1, cv2.LINE_AA)


# ── main class ────────────────────────────────────────────────────────────

class LipTargetingSystem:
    """
    Wraps a webcam + MediaPipe FaceDetector to extract faces.
    """

    def __init__(self, camera_index: int | None = None) -> None:
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        
        self._detector = None

        # thread-safe target store
        self._lock = threading.Lock()
        self._current_angle: float | None = None
        self._current_lip_crop: np.ndarray | None = None
        self._current_face_crop: np.ndarray | None = None
        self._current_raw_frame: np.ndarray | None = None

        # Lock-on tracking
        self.is_locked: bool = False
        self._locked_nose_pos: Tuple[float, float] | None = None

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "LipTargetingSystem":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        _ensure_model()
        
        if self._camera_index is not None:
            self._cap = cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self._camera_index}")
        else:
            # Auto-detect: try index 1 (usually external USB webcam) then 0 (internal)
            self._cap = cv2.VideoCapture(1)
            if self._cap.isOpened():
                print("[Camera] Auto-selected external camera (index 1)")
            else:
                self._cap = cv2.VideoCapture(0)
                if self._cap.isOpened():
                    print("[Camera] Auto-selected internal camera (index 0)")
                else:
                    raise RuntimeError("Could not open any camera (tried 1 and 0)")
        
        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._detector = None
        cv2.destroyAllWindows()

    # ── public properties ─────────────────────────────────────────────────

    @property
    def current_target_data(self) -> Tuple[float | None, np.ndarray | None, np.ndarray | None]:
        """Returns (angle, lip_crop_frame, face_crop_frame) from the most recent capture."""
        with self._lock:
            return self._current_angle, self._current_lip_crop, self._current_face_crop

    @property
    def current_raw_frame(self) -> np.ndarray | None:
        """Returns the latest raw BGR camera frame (full resolution)."""
        with self._lock:
            return self._current_raw_frame

    # ── generator interface (integration-ready) ───────────────────────────

    def stream(self) -> Generator[Tuple[float | None, np.ndarray | None, np.ndarray | None], None, None]:
        for _ in self._run_capture_loop(display=False):
            yield self.current_target_data

    # ── internal capture loop ─────────────────────────────────────────────

    def _get_face_bounding_box(self, detection, h, w) -> Tuple[int, int, int, int]:
        """Calculates a slightly padded bounding box around the full face."""
        bb = detection.bounding_box
        x_min = int(bb.origin_x)
        y_min = int(bb.origin_y)
        x_max = int(bb.origin_x + bb.width)
        y_max = int(bb.origin_y + bb.height)
        
        # Add padding (e.g., 20% to capture full head context)
        pad_x = int(bb.width * 0.2)
        pad_y = int(bb.height * 0.2)
        
        return (
            max(0, x_min - pad_x),
            max(0, y_min - pad_y),
            min(w - 1, x_max + pad_x),
            min(h - 1, y_max + pad_y)
        )

    def _run_capture_loop(self, display: bool = True) -> Iterator[None]:
        assert self._cap is not None and self._detector is not None
        
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break

            angle, lip_crop, face_crop = self._process_frame(frame, draw=display)

            with self._lock:
                self._current_angle = angle
                self._current_lip_crop = lip_crop
                self._current_face_crop = face_crop
                self._current_raw_frame = frame.copy()

            if display:
                cv2.imshow("AV-TSE Targeting (Face Tracker)", frame)
                if face_crop is not None:
                    cv2.imshow("Extracted Face to network", face_crop)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:   # q or Esc to quit
                    break

            yield   # hand control back (allows stream() to yield data)

    def _process_frame(self, frame, draw: bool = True) -> Tuple[float | None, np.ndarray | None, np.ndarray | None]:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(mp_image)

        if draw:
            _draw_crosshair(frame, cx, cy)

        if not results.detections:
            return None, None, None

        # ── find target ──────────────────────────────────────────────────
        if self.is_locked and self._locked_nose_pos is not None:
            # locked: find face whose nose is closest to the last known position
            best_face = min(
                results.detections,
                key=lambda det: (det.keypoints[_NOSE_TIP_IDX].x - self._locked_nose_pos[0])**2 +
                                 (det.keypoints[_NOSE_TIP_IDX].y - self._locked_nose_pos[1])**2
            )
        else:
            # unlocked: face whose nose_x is closest to 0.5
            best_face = min(
                results.detections,
                key=lambda det: abs(det.keypoints[_NOSE_TIP_IDX].x - CENTER_X),
            )
        
        # update tracked position
        self._locked_nose_pos = (best_face.keypoints[_NOSE_TIP_IDX].x, best_face.keypoints[_NOSE_TIP_IDX].y)

        target_angle: float | None = None
        lip_crop: np.ndarray | None = None
        face_crop: np.ndarray | None = None

        for det in results.detections:
            is_target = det is best_face
            color = COLOR_TARGET if is_target else COLOR_BG_FACE
            thickness = 2 if is_target else 1

            nose_x = det.keypoints[_NOSE_TIP_IDX].x
            nose_y = det.keypoints[_NOSE_TIP_IDX].y
            nx, ny = int(nose_x * w), int(nose_y * h)

            # Draw nose dot
            if draw:
                cv2.circle(frame, (nx, ny), 4 if is_target else 2, color, -1, cv2.LINE_AA)

            if is_target:
                target_angle = _angle_from_nose_x(nose_x)

                # Extract face crop
                fx1, fy1, fx2, fy2 = self._get_face_bounding_box(det, h, w)
                if fx2 > fx1 and fy2 > fy1:
                    raw_face_crop = frame[fy1:fy2, fx1:fx2]
                    try:
                        face_crop = cv2.resize(raw_face_crop, CROP_SIZE)
                    except Exception:
                        pass
                
                # Ensure valid bounding box crop
                if draw and fx1 != fx2:
                    # Draw box around face
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), COLOR_TARGET, 2)
                    sign = "+" if target_angle >= 0 else ""
                    if self.is_locked:
                        label = f"LOCKED: {sign}{target_angle:.1f} deg"
                    else:
                        label = f"Target: {sign}{target_angle:.1f} deg"
                    cv2.putText(
                        frame, label,
                        (max(0, fx1), max(fy1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR_TARGET, 2, cv2.LINE_AA,
                    )

        return target_angle, lip_crop, face_crop


# ── standalone entry-point ────────────────────────────────────────────────

def main() -> None:
    print("AV-TSE Target Setup  |  press Q or Esc to quit")
    try:
        with LipTargetingSystem(camera_index=None) as ts:
            for _ in ts._run_capture_loop(display=True):
                angle, lip_crop, face_crop = ts.current_target_data
                if angle is not None and face_crop is not None:
                    sign = "+" if angle >= 0 else ""
                    print(f"\rTarget Tracker Locked: {sign}{angle:6.1f} °   | Yielding {face_crop.shape} shape crops to network...", end="", flush=True)
                else:
                    print("\rSearching for Target...                                                          ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()


if __name__ == "__main__":
    main()
