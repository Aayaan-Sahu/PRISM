"""
Targeting System (AV-TSE) — Extracts target angle and cropped lips for neural speech extraction.

Upgraded to use MediaPipe FaceMesh to tightly bound the lips of the face closest to the
horizontal target crosshair. 

Yields (target_angle, lip_crop_bgr) on every frame.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from typing import Generator, Iterator, Tuple

import cv2
import numpy as np
import mediapipe as mp

# ── constants ──────────────────────────────────────────────────────────────
HFOV_DEG: float = 60.0          # horizontal field-of-view of webcam
CENTER_X: float = 0.5           # normalised horizontal centre
CROSSHAIR_HALF: int = 20        # pixels for static crosshair arms

COLOR_TARGET   = (0,   255,   0)   # bright green
COLOR_BG_FACE  = (160, 160, 160)   # grey
COLOR_CROSS    = (255, 255, 255)   # white

# MediaPipe FaceMesh canonical nose tip
_NOSE_TIP_IDX = 1

# MediaPipe FaceMesh outer lip indices 
# (roughly forming the convex hull around the mouth)
_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

# Neural network expected lip crop size
CROP_SIZE = (96, 96) 


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
    Wraps a webcam + MediaPipe FaceMesh to specifically extract lips.

    Thread-safe logic allows an audio inference thread to continuously
    poll `current_target_data` without blocking.
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = None

        # thread-safe target store
        self._lock = threading.Lock()
        self._current_angle: float | None = None
        self._current_lip_crop: np.ndarray | None = None
        self._current_raw_frame: np.ndarray | None = None

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "LipTargetingSystem":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_index}")
        
        self._face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        if self._face_mesh is not None:
            self._face_mesh.close()
        cv2.destroyAllWindows()

    # ── public properties ─────────────────────────────────────────────────

    @property
    def current_target_data(self) -> Tuple[float | None, np.ndarray | None]:
        """Returns (angle, lip_crop_frame) from the most recent capture."""
        with self._lock:
            return self._current_angle, self._current_lip_crop

    @property
    def current_raw_frame(self) -> np.ndarray | None:
        """Returns the latest raw BGR camera frame (full resolution)."""
        with self._lock:
            return self._current_raw_frame

    # ── generator interface (integration-ready) ───────────────────────────

    def stream(self) -> Generator[Tuple[float | None, np.ndarray | None], None, None]:
        for _ in self._run_capture_loop(display=False):
            yield self.current_target_data

    # ── internal capture loop ─────────────────────────────────────────────

    def _get_lip_bounding_box(self, face_landmarks, h, w) -> Tuple[int, int, int, int]:
        """Calculates a slightly padded bounding box around the lips."""
        xs = [int(face_landmarks.landmark[i].x * w) for i in _LIP_INDICES]
        ys = [int(face_landmarks.landmark[i].y * h) for i in _LIP_INDICES]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Add padding (e.g., 20% to capture full articulation context)
        pad_x = int((x_max - x_min) * 0.2)
        pad_y = int((y_max - y_min) * 0.2)
        
        return (
            max(0, x_min - pad_x),
            max(0, y_min - pad_y),
            min(w - 1, x_max + pad_x),
            min(h - 1, y_max + pad_y)
        )

    def _run_capture_loop(self, display: bool = True) -> Iterator[None]:
        assert self._cap is not None and self._face_mesh is not None
        
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break

            angle, lip_crop = self._process_frame(frame, draw=display)

            with self._lock:
                self._current_angle = angle
                self._current_lip_crop = lip_crop
                self._current_raw_frame = frame.copy()

            if display:
                cv2.imshow("AV-TSE Targeting (Lip Tracker)", frame)
                if lip_crop is not None:
                    cv2.imshow("Extracted Lips to network", lip_crop)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:   # q or Esc to quit
                    break

            yield   # hand control back (allows stream() to yield data)

    def _process_frame(self, frame, draw: bool = True) -> Tuple[float | None, np.ndarray | None]:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if draw:
            _draw_crosshair(frame, cx, cy)

        if not results.multi_face_landmarks:
            return None, None

        # ── find target: face whose nose_x is closest to 0.5 ─────────────
        best_face = min(
            results.multi_face_landmarks,
            key=lambda face: abs(face.landmark[_NOSE_TIP_IDX].x - CENTER_X),
        )

        target_angle: float | None = None
        lip_crop: np.ndarray | None = None

        for face_landmarks in results.multi_face_landmarks:
            is_target = face_landmarks is best_face
            color = COLOR_TARGET if is_target else COLOR_BG_FACE
            thickness = 2 if is_target else 1

            nose_x = face_landmarks.landmark[_NOSE_TIP_IDX].x
            nose_y = face_landmarks.landmark[_NOSE_TIP_IDX].y
            nx, ny = int(nose_x * w), int(nose_y * h)

            # Draw nose dot
            if draw:
                cv2.circle(frame, (nx, ny), 4 if is_target else 2, color, -1, cv2.LINE_AA)

            if is_target:
                target_angle = _angle_from_nose_x(nose_x)
                
                # Extract lip crop
                lx1, ly1, lx2, ly2 = self._get_lip_bounding_box(face_landmarks, h, w)
                
                # Ensure valid bounding box crop
                if lx2 > lx1 and ly2 > ly1:
                    raw_crop = frame[ly1:ly2, lx1:lx2]
                    try:
                        # Resize to uniform shape (96x96) for Neural Net
                        lip_crop = cv2.resize(raw_crop, CROP_SIZE)
                    except Exception:
                        pass # Ignore edge-case zero-sized crops near frame boundaries

                if draw and lx1 != lx2:
                    # Draw box around lips
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_TARGET, 2)
                    sign = "+" if target_angle >= 0 else ""
                    label = f"Target: {sign}{target_angle:.1f} deg"
                    cv2.putText(
                        frame, label,
                        (lx1, max(ly1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLOR_TARGET, 2, cv2.LINE_AA,
                    )

        return target_angle, lip_crop


# ── standalone entry-point ────────────────────────────────────────────────

def main() -> None:
    print("AV-TSE Target Setup  |  press Q or Esc to quit")
    try:
        with LipTargetingSystem(camera_index=0) as ts:
            for _ in ts._run_capture_loop(display=True):
                angle, crop = ts.current_target_data
                if angle is not None and crop is not None:
                    sign = "+" if angle >= 0 else ""
                    print(f"\rTarget Tracker Locked: {sign}{angle:6.1f} °   | Yielding {crop.shape} shape crops to network...", end="", flush=True)
                else:
                    print("\rSearching for Target...                                                          ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()


if __name__ == "__main__":
    main()
