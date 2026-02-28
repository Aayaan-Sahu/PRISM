"""
Face Lock — re-identification tracker for locking onto a specific face.

Uses spatial proximity with velocity prediction and scale-invariant keypoint
descriptors to maintain a lock on a target face across frames.

States:  UNLOCKED → LOCKED → LOST
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field

import numpy as np


# MediaPipe face detection keypoint indices
_KP_RIGHT_EYE = 0
_KP_LEFT_EYE = 1
_KP_NOSE_TIP = 2
_KP_MOUTH_CENTRE = 3
_KP_RIGHT_EAR = 4
_KP_LEFT_EAR = 5
_NUM_KP = 6

# Number of pairwise distances: C(6,2) = 15
_NUM_PAIRS = _NUM_KP * (_NUM_KP - 1) // 2

# Thresholds
_IOU_THRESHOLD = 0.3
_DESCRIPTOR_THRESHOLD = 0.85
_MAX_DISTANCE_PX = 150       # hard cap: reject candidates farther than this
_LOST_TIMEOUT_FRAMES = 15    # ~0.5s at 30 FPS


class LockState(enum.Enum):
    UNLOCKED = "UNLOCKED"
    LOCKED = "LOCKED"
    LOST = "LOST"


@dataclass
class FaceSnapshot:
    """Snapshot of a detected face for re-identification."""

    # Bounding box in pixel coords
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

    # Centre of bounding box (pixels)
    cx: float
    cy: float

    # Scale-invariant keypoint descriptor (15 pairwise distances, normalised)
    descriptor: np.ndarray  # shape (15,)

    @staticmethod
    def from_detection(detection, frame_h: int, frame_w: int) -> FaceSnapshot:
        """Build a snapshot from a MediaPipe detection object."""
        bb = detection.bounding_box
        bbox_x = float(bb.origin_x)
        bbox_y = float(bb.origin_y)
        bbox_w = float(bb.width)
        bbox_h = float(bb.height)
        cx = bbox_x + bbox_w / 2.0
        cy = bbox_y + bbox_h / 2.0

        descriptor = _build_descriptor(detection, frame_h, frame_w)

        return FaceSnapshot(
            bbox_x=bbox_x, bbox_y=bbox_y,
            bbox_w=bbox_w, bbox_h=bbox_h,
            cx=cx, cy=cy,
            descriptor=descriptor,
        )


def _build_descriptor(detection, frame_h: int, frame_w: int) -> np.ndarray:
    """
    Build a 15-element scale-invariant descriptor from the 6 MediaPipe keypoints.

    Each element is the pairwise Euclidean distance between two keypoints,
    normalised by the bounding-box diagonal so the descriptor is scale-invariant.
    """
    bb = detection.bounding_box
    diag = np.sqrt(bb.width ** 2 + bb.height ** 2)
    if diag < 1e-6:
        return np.zeros(_NUM_PAIRS, dtype=np.float32)

    # Extract keypoint pixel positions
    pts = np.array(
        [(kp.x * frame_w, kp.y * frame_h) for kp in detection.keypoints[:_NUM_KP]],
        dtype=np.float32,
    )

    # Compute all pairwise distances
    pairs = []
    for i in range(_NUM_KP):
        for j in range(i + 1, _NUM_KP):
            d = np.linalg.norm(pts[i] - pts[j])
            pairs.append(d / diag)

    return np.array(pairs, dtype=np.float32)


def _iou(a: FaceSnapshot, bx: float, by: float, bw: float, bh: float) -> float:
    """Intersection over Union between snapshot bbox and raw bbox."""
    ax1, ay1 = a.bbox_x, a.bbox_y
    ax2, ay2 = a.bbox_x + a.bbox_w, a.bbox_y + a.bbox_h
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax1, bx)
    iy1 = max(ay1, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = a.bbox_w * a.bbox_h
    area_b = bw * bh
    union = area_a + area_b - inter

    if union < 1e-6:
        return 0.0
    return inter / union


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class FaceLock:
    """
    Tracks a locked face across frames using spatial + descriptor matching.

    Usage::

        lock = FaceLock()
        lock.lock(snapshot)          # lock onto a face
        idx = lock.match(snapshots)  # each frame, find which detection is the target
        lock.unlock()                # release
    """

    def __init__(self) -> None:
        self._state = LockState.UNLOCKED
        self._target: FaceSnapshot | None = None
        self._lost_count: int = 0

        # Velocity prediction: store last N centres
        self._history: deque[tuple[float, float]] = deque(maxlen=3)

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def target(self) -> FaceSnapshot | None:
        return self._target

    def lock(self, snapshot: FaceSnapshot) -> None:
        """Lock onto the given face."""
        self._target = snapshot
        self._state = LockState.LOCKED
        self._lost_count = 0
        self._history.clear()
        self._history.append((snapshot.cx, snapshot.cy))

    def unlock(self) -> None:
        """Release the lock."""
        self._state = LockState.UNLOCKED
        self._target = None
        self._lost_count = 0
        self._history.clear()

    def _predicted_centre(self) -> tuple[float, float]:
        """Predict next position using linear velocity from recent history."""
        if len(self._history) < 2:
            return self._history[-1] if self._history else (0.0, 0.0)

        # Average velocity over history
        pts = list(self._history)
        vx = (pts[-1][0] - pts[0][0]) / len(pts)
        vy = (pts[-1][1] - pts[0][1]) / len(pts)
        return (pts[-1][0] + vx, pts[-1][1] + vy)

    def match(self, candidates: list[FaceSnapshot]) -> int | None:
        """
        Find the locked face among candidate detections.

        Returns the index of the best match, or None if no match found.
        Transitions to LOST after too many consecutive misses.
        """
        if self._state == LockState.UNLOCKED:
            return None
        if self._target is None:
            return None
        if not candidates:
            self._lost_count += 1
            if self._lost_count >= _LOST_TIMEOUT_FRAMES:
                self._state = LockState.LOST
            return None

        pred_cx, pred_cy = self._predicted_centre()

        best_idx: int | None = None
        best_score: float = -1.0

        for i, cand in enumerate(candidates):
            # Hard distance cap — reject faces too far from predicted position
            dist = np.sqrt((cand.cx - pred_cx) ** 2 + (cand.cy - pred_cy) ** 2)
            if dist > _MAX_DISTANCE_PX:
                continue

            # Compute IoU with predicted position
            pred_snap = FaceSnapshot(
                bbox_x=pred_cx - self._target.bbox_w / 2,
                bbox_y=pred_cy - self._target.bbox_h / 2,
                bbox_w=self._target.bbox_w,
                bbox_h=self._target.bbox_h,
                cx=pred_cx, cy=pred_cy,
                descriptor=self._target.descriptor,
            )
            iou_val = _iou(pred_snap, cand.bbox_x, cand.bbox_y, cand.bbox_w, cand.bbox_h)

            # Compute descriptor similarity
            desc_sim = _cosine_similarity(self._target.descriptor, cand.descriptor)

            # Require spatial overlap OR strong descriptor match,
            # but always require minimum descriptor similarity to avoid
            # locking onto a completely different face nearby
            if desc_sim < 0.7:
                continue
            if iou_val < _IOU_THRESHOLD and desc_sim < _DESCRIPTOR_THRESHOLD:
                continue

            # Combined score: spatial proximity (primary) + descriptor (secondary)
            dist_score = max(0.0, 1.0 - dist / _MAX_DISTANCE_PX)
            score = 0.7 * dist_score + 0.3 * desc_sim

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            # Update target with latest detection
            self._target = candidates[best_idx]
            self._history.append((candidates[best_idx].cx, candidates[best_idx].cy))
            self._lost_count = 0
            if self._state == LockState.LOST:
                self._state = LockState.LOCKED
        else:
            self._lost_count += 1
            if self._lost_count >= _LOST_TIMEOUT_FRAMES:
                self._state = LockState.LOST

        return best_idx
