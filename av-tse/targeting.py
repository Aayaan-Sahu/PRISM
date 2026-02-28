"""
Targeting System — real-time face tracker for audio beamformer.

Detects all faces via MediaPipe, locks onto the face whose nose tip is
closest to the horizontal centre of the frame, and yields the azimuth
angle in degrees.

  -30 °  ←  left edge       0 °  ← centre       +30 °  →  right edge

Face lock:  press L to lock onto the current target and start a 5-second
recording.  Press L again to unlock early.

Usage (standalone):
    python targeting.py

Usage (integration) — import and iterate:
    from targeting import TargetingSystem
    with TargetingSystem() as ts:
        for angle in ts.stream():
            audio_thread.set_angle(angle)
"""

from __future__ import annotations

import math
import os
import threading
import urllib.request
from contextlib import contextmanager
from typing import Generator, Iterator

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from face_lock import FaceLock, FaceSnapshot, LockState
from recorder import AVRecorder, save_clip
from dolphin_preprocess import preprocess_for_dolphin

# ── model ──────────────────────────────────────────────────────────────────
_MODEL_PATH = "../blaze_face_short_range.tflite"
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
COLOR_LOCKED   = (255, 255,   0)   # cyan (BGR)
COLOR_REC      = (0,     0, 255)   # red (BGR)
COLOR_LOST     = (0,     0, 255)   # red (BGR)

# Output directory for recordings
_RECORDINGS_DIR = os.path.join(".", "recordings", "latest")


# ── MediaPipe nose-tip keypoint index ─────────────────────────────────────
# face_detection keypoints order:
#  0 right_eye  1 left_eye  2 nose_tip  3 mouth_centre
#  4 right_ear  5 left_ear
_NOSE_TIP_IDX = 2


def _angle_from_nose_x(nose_x: float) -> float:
    """Map normalised nose x-coordinate → azimuth angle in degrees."""
    return (nose_x - CENTER_X) * HFOV_DEG


def _draw_crosshair(frame, cx: int, cy: int) -> None:
    h = CROSSHAIR_HALF
    cv2.line(frame, (cx - h, cy), (cx + h, cy), COLOR_CROSS, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - h), (cx, cy + h), COLOR_CROSS, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, COLOR_CROSS, -1, cv2.LINE_AA)


def _draw_bbox(frame, detection, h: int, w: int, color, thickness: int = 2) -> None:
    bb = detection.bounding_box  # pixel coordinates in Tasks API
    x1 = max(0, int(bb.origin_x))
    y1 = max(0, int(bb.origin_y))
    x2 = min(w - 1, int(bb.origin_x + bb.width))
    y2 = min(h - 1, int(bb.origin_y + bb.height))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return x1, y1, x2, y2


# ── main class ────────────────────────────────────────────────────────────

class TargetingSystem:
    """
    Wraps a webcam + MediaPipe face detector.

    Thread-safe: `current_angle` is updated by the capture loop and can
    be read from any thread without blocking.

    Context-manager usage releases the camera automatically::

        with TargetingSystem() as ts:
            for angle in ts.stream():
                ...
    """

    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._detector = None

        # thread-safe angle store
        self._lock = threading.Lock()
        self._current_angle: float | None = None

        # face lock-on
        self._face_lock = FaceLock()

        # A/V recorder
        self._recorder = AVRecorder()

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "TargetingSystem":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def open(self) -> None:
        _ensure_model()
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_index}")
        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def close(self) -> None:
        if self._recorder.is_recording:
            self._recorder.cancel()
        if self._cap is not None:
            self._cap.release()
        self._detector = None
        cv2.destroyAllWindows()

    # ── public angle property ─────────────────────────────────────────────

    @property
    def current_angle(self) -> float | None:
        """Most-recent target angle in degrees; None if no face detected."""
        with self._lock:
            return self._current_angle

    # ── generator interface (integration-ready) ───────────────────────────

    def stream(self) -> Generator[float | None, None, None]:
        """
        Yields the current target angle on every processed frame.

        Suitable for driving an audio thread::

            for angle in ts.stream():
                if angle is not None:
                    beamformer.steer(angle)
        """
        for _ in self._run_capture_loop(display=False):
            yield self.current_angle

    # ── key handling ──────────────────────────────────────────────────────

    def _handle_key(self, key: int, detections, frame_h: int, frame_w: int) -> bool:
        """Handle keyboard input. Returns True if should quit."""
        if key == ord("q") or key == 27:  # q or Esc
            return True

        if key == ord("l") or key == ord("L"):
            if self._face_lock.state == LockState.UNLOCKED:
                # Lock onto current target and start recording
                self._try_lock_and_record(detections, frame_h, frame_w)
            else:
                # Unlock and stop recording
                self._unlock_and_stop()

        return False

    def _try_lock_and_record(self, detections, frame_h: int, frame_w: int) -> None:
        """Lock onto the closest-to-center face and start recording."""
        if not detections:
            return

        # Find closest to center
        best = min(
            detections,
            key=lambda d: abs(d.keypoints[_NOSE_TIP_IDX].x - CENTER_X),
        )
        snapshot = FaceSnapshot.from_detection(best, frame_h, frame_w)
        self._face_lock.lock(snapshot)
        self._recorder.start()
        print("\n[LOCK] Locked onto face — recording 5 seconds...")

    def _unlock_and_stop(self) -> None:
        """Unlock face and stop recording."""
        self._face_lock.unlock()
        if self._recorder.is_recording:
            clip = self._recorder.finish()
            print("\n[UNLOCK] Early unlock — saving partial recording...")
            self._save_recording(clip)
        else:
            print("\n[UNLOCK] Released lock.")

    def _save_recording(self, clip) -> None:
        """Save clip and run Dolphin preprocessing in background thread."""
        def _save():
            raw_dir = os.path.join(_RECORDINGS_DIR, "raw")
            dolphin_dir = os.path.join(_RECORDINGS_DIR, "dolphin")
            save_clip(clip, raw_dir)
            print(f"[SAVE] Raw clip saved to {raw_dir}")
            preprocess_for_dolphin(clip, dolphin_dir)
            print(f"[SAVE] Dolphin-ready output saved to {dolphin_dir}")

        t = threading.Thread(target=_save, daemon=True)
        t.start()

    # ── internal capture loop ─────────────────────────────────────────────

    def _run_capture_loop(self, display: bool = True) -> Iterator[None]:
        assert self._cap is not None and self._detector is not None, \
            "Call open() first (or use as a context manager)."

        while True:
            ok, frame = self._cap.read()
            if not ok:
                break

            angle = self._process_frame(frame, draw=display)

            with self._lock:
                self._current_angle = angle

            if display:
                cv2.imshow("Targeting System", frame)
                key = cv2.waitKey(1) & 0xFF

                # Gather current detections for key handler
                # (re-use cached detections from _process_frame)
                if self._handle_key(key, self._last_detections, *frame.shape[:2]):
                    break

            yield   # hand control back (allows stream() to yield angle)

    def _process_frame(self, frame, draw: bool = True) -> float | None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(mp_image)

        if draw:
            _draw_crosshair(frame, cx, cy)

        detections = results.detections if results.detections else []
        self._last_detections = detections  # cache for key handler

        if not detections:
            # No faces — update lock state
            if self._face_lock.state == LockState.LOCKED:
                self._face_lock.match([])  # increment lost counter

                # Check if lock was just lost
                if self._face_lock.state == LockState.LOST:
                    self._on_lock_lost(draw, frame, h, w)

            if draw and self._face_lock.state == LockState.LOST:
                cv2.putText(frame, "LOCK LOST", (cx - 60, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LOST, 2, cv2.LINE_AA)

            return None

        # ── determine target face ─────────────────────────────────────────
        target_det = None
        is_locked = self._face_lock.state in (LockState.LOCKED, LockState.LOST)

        if is_locked:
            # Build snapshots for all candidates
            snapshots = [FaceSnapshot.from_detection(d, h, w) for d in detections]
            match_idx = self._face_lock.match(snapshots)

            if match_idx is not None:
                target_det = detections[match_idx]
            elif self._face_lock.state == LockState.LOST:
                self._on_lock_lost(draw, frame, h, w)
        else:
            # Default: closest to center
            target_det = min(
                detections,
                key=lambda d: abs(d.keypoints[_NOSE_TIP_IDX].x - CENTER_X),
            )

        # ── draw all faces + compute angle ────────────────────────────────
        target_angle: float | None = None

        for det in detections:
            is_target = det is target_det

            if is_target and self._face_lock.state == LockState.LOCKED:
                color = COLOR_LOCKED
                thickness = 3
            elif is_target:
                color = COLOR_TARGET
                thickness = 2
            else:
                color = COLOR_BG_FACE
                thickness = 1

            kp = det.keypoints[_NOSE_TIP_IDX]
            nose_x, nose_y = kp.x, kp.y
            nx, ny = int(nose_x * w), int(nose_y * h)

            if draw:
                x1, y1, x2, y2 = _draw_bbox(frame, det, h, w, color, thickness)

            if is_target:
                target_angle = _angle_from_nose_x(nose_x)
                sign = "+" if target_angle >= 0 else ""

                if self._face_lock.state == LockState.LOCKED:
                    label = f"LOCKED: {sign}{target_angle:.1f} deg"
                else:
                    label = f"Target: {sign}{target_angle:.1f} deg"

                if draw:
                    cv2.circle(frame, (nx, ny), 5, color, -1, cv2.LINE_AA)
                    cv2.putText(
                        frame, label,
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA,
                    )
            else:
                if draw:
                    cv2.circle(frame, (nx, ny), 3, COLOR_BG_FACE, -1, cv2.LINE_AA)

        # ── recording ─────────────────────────────────────────────────────
        if self._recorder.is_recording:
            # Push frame to recorder (with locked face's detection)
            self._recorder.push_frame(frame, target_det)

            if draw:
                self._draw_rec_indicator(frame, w)

            # Check if recording is complete
            if self._recorder.is_done:
                clip = self._recorder.finish()
                self._face_lock.unlock()
                print("\n[REC] 5-second recording complete!")
                self._save_recording(clip)

        # Show LOCK LOST overlay
        if draw and self._face_lock.state == LockState.LOST:
            cv2.putText(frame, "LOCK LOST", (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_LOST, 2, cv2.LINE_AA)

        return target_angle

    def _draw_rec_indicator(self, frame, frame_w: int) -> None:
        """Draw red REC dot and elapsed time in top-right corner."""
        elapsed = self._recorder.elapsed
        remaining = max(0.0, 5.0 - elapsed)

        # Red circle (REC dot)
        dot_x = frame_w - 80
        dot_y = 30
        cv2.circle(frame, (dot_x, dot_y), 8, COLOR_REC, -1, cv2.LINE_AA)

        # Time text
        text = f"REC {remaining:.1f}s"
        cv2.putText(frame, text, (dot_x + 14, dot_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_REC, 2, cv2.LINE_AA)

    def _on_lock_lost(self, draw: bool, frame, h: int, w: int) -> None:
        """Handle transition to LOST state."""
        if self._recorder.is_recording:
            clip = self._recorder.finish()
            print("\n[LOST] Lock lost — saving partial recording...")
            self._save_recording(clip)
            self._face_lock.unlock()


# ── standalone entry-point ────────────────────────────────────────────────

def main() -> None:
    print("Targeting System  |  L = lock/unlock + record  |  Q/Esc = quit")
    try:
        with TargetingSystem(camera_index=0) as ts:
            for _ in ts._run_capture_loop(display=True):
                angle = ts.current_angle
                if angle is not None:
                    sign = "+" if angle >= 0 else ""
                    print(f"\rTarget angle: {sign}{angle:6.1f} °   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()


if __name__ == "__main__":
    main()