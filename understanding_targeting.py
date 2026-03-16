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

_DETECTOR_MODEL_PATH = "blaze_face_short_range.tflite"
_DETECTOR_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"

_LANDMARKER_MODEL_PATH = "face_landmarker.task"
_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

COLOR_TARGET = (0, 255, 0)
COLOR_BG_FACE = (160, 160, 160)
COLOR_CROSS = (255, 255, 255)

CROSSHAIR_HALF = 20

_NOSE_TIP_IDX = 2

CROP_SIZE = (512, 512)

def _ensure_models() -> None:
    if not os.path.exists(_DETECTOR_MODEL_PATH):
        print(f"Downloading detector model to {_DETECTOR_MODEL_PATH}...")
        urllib.request.urlretrieve(_DETECTOR_MODEL_URL, _DETECTOR_MODEL_PATH)
    if not os.path.exists(_LANDMARKER_MODEL_PATH):
        print(f"Downloading landmarker model to {_LANDMARKER_MODEL_PATH}...")
        urllib.request.urlretrieve(_LANDMARKER_MODEL_URL, _LANDMARKER_MODEL_PATH)
        print("Downloads complete.")

def _angle_from_nose_x(nose_x: float) -> float:
    return (nose_x - 0.5) * 2 * 30

def _draw_crosshair(frame, cx: int, cy: int) -> None:
    h = CROSSHAIR_HALF
    cv2.line(frame, (cx - h, cy), (cx + h, cy), COLOR_CROSS, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - h), (cx, cy + h), COLOR_CROSS, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, COLOR_CROSS, -1, cv2.LINE_AA)

class LipTargetingSystem:
    def __init__(self, camera_index: int | None = None) -> None:
        self._camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

        self._detector = None

        # threading safe target store
        self._lock = threading.Lock()
        self._current_angle: float | None = None
        self._current_lip_crop: np.ndarray | None = None
        self._current_face_crop: np.ndarray | None = None
        self._current_raw_frame: np.ndarray | None = None

        self.is_locked: bool = False
        self.is_recording: bool = False
        self._locked_face_id: int | None = None
        
        # EMA Smoothing state
        self._ema_landmarks = None
        self._ema_scale = None
        self._ema_angle = None
        
        self._alpha = 0.3  # Alpha for landmarks
        self._alpha_slow = 0.1 # Alpha for scale/rotation to make it very steady

    def __enter__(self) -> "LipTargetingSystem":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def open(self) -> None:
        # 1. Ensure face detection model is installed
        # 2. Open the camera
        # 3. Initialize the face detector

        # Check whether face detection model is available, and download if not
        _ensure_models()

        # Open the camera
        if self._camera_index is not None:
            self._capture = cv2.VideoCapture(self._camera_index)
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not open camera with index {self._camera_index}")
        else:
            self._capture = cv2.VideoCapture(1)
            if self._capture.isOpened():
                print("[CAMERA] Auto-selected external camera (index 1)")
            else:
                self._capture = cv2.VideoCapture(0)
                if self._capture.isOpened():
                    print("[CAMERA] Auto-selected default camera (index 0)")
                else:
                    raise RuntimeError("Could not open any camera")
        
        # Initialize the face landmarker
        base_options = mp_python.BaseOptions(model_asset_path=_LANDMARKER_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self._detector = None
        cv2.destroyAllWindows()

    @property
    def current_target_data(self) -> Tuple[float | None, np.ndarray | None, np.ndarray | None]:
        # Returns the current lip targeting data as a tuple of (angle, lip_crop, face_crop)
        with self._lock:
            return self._current_angle, self._current_lip_crop, self._current_face_crop

    @property
    def current_raw_frame(self) -> np.ndarray | None:
        # Returns the latest raw BGR camera frame
        with self._lock:
            return self._current_raw_frame

    def stream(self, display: bool = True) -> Generator[Tuple[float | None, np.ndarray | None, np.ndarray | None], None, None]:
        # A generator that yields the current targeting data whenever a new frame is processed. This can be used to drive the rest of the system.
        for _ in self._run_capture_loop(display=display):
            yield self.current_target_data


    def _run_capture_loop(self, display: bool = True) -> Iterator[None]:
        assert self._capture is not None and self._detector is not None

        while True:
            # Read a frame from the camera
            ok, frame = self._capture.read()
            if not ok:
                break

            angle, lip_crop, face_crop = self._process_frame(frame, draw=display)

            # thread-safely update the current targeting data
            with self._lock:
                self._current_angle = angle
                self._current_lip_crop = lip_crop
                self._current_face_crop = face_crop
                self._current_raw_frame = frame.copy()
            
            # Display the frame if requested
            if display:
                cv2.imshow("Face Tracker", frame)
                if face_crop is not None:
                    cv2.imshow("Extracted Face to network", face_crop)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                # Toggle lock with 'l' key
                elif key == ord("l"):
                    self.is_locked = not self.is_locked
                    if not self.is_locked:
                        self._locked_nose_position = None
                        print("\n[TARGETING] Unlocked from face.")
                    else:
                        print("\n[TARGETING] Locked onto face.")
                elif key == ord("i"):
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        print("\n[TARGETING] Recording STARTED.")
                    else:
                        print("\n[TARGETING] Recording STOPPED.")
            
            # hand control back to caller to do other work if needed
            yield

    def _align_face(self, frame, landmarks, target_size=(512, 512)) -> np.ndarray:
        h, w = frame.shape[:2]
        # Get coordinates for eyes and nose
        # Indices in FaceMesh: Left Outer Eye: 33, Right Outer Eye: 263, Nose: 1
        l_eye = np.array([landmarks[33].x * w, landmarks[33].y * h])
        r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
        nose = np.array([landmarks[1].x * w, landmarks[1].y * h])

        # Smooth landmarks using EMA
        if self._ema_landmarks is None:
            self._ema_landmarks = landmarks
        else:
            # We only smooth the specific points used for alignment to keep it simple but effective
            pass 

        # Calculate angle for rotation
        dy = r_eye[1] - l_eye[1]
        dx = r_eye[0] - l_eye[0]
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Distance between eyes
        eye_dist = np.sqrt(dx**2 + dy**2)
        
        # We want the face to occupy a specific resolution and be centered
        # Zoomed out: eyes should be ~22% of the way down, and dist should be ~22% of width
        desired_eye_dist = target_size[0] * 0.22
        raw_scale = desired_eye_dist / eye_dist

        # Smooth scale and angle to prevent zooming in when turning head (yaw)
        if self._ema_scale is None:
            self._ema_scale = raw_scale
            self._ema_angle = angle_deg
        else:
            self._ema_scale = self._alpha_slow * raw_scale + (1 - self._alpha_slow) * self._ema_scale
            self._ema_angle = self._alpha_slow * angle_deg + (1 - self._alpha_slow) * self._ema_angle

        # Center of rotation (between eyes)
        center = (l_eye + r_eye) / 2
        
        # Transformation matrix using smoothed values
        M = cv2.getRotationMatrix2D(tuple(center), self._ema_angle, self._ema_scale)

        # Shift to center the face in the target image
        # Zoomed out values: Center (0.5, 0.45)
        tx = target_size[0] * 0.5 - center[0]
        ty = target_size[1] * 0.45 - center[1]
        
        M[0, 2] += tx
        M[1, 2] += ty

        aligned = cv2.warpAffine(frame, M, target_size, flags=cv2.INTER_LANCZOS4)
        return aligned

    def _process_frame(self, frame, draw: bool = True) -> Tuple[float | None, np.ndarray | None, np.ndarray | None]:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Use timestamp in ms for video mode
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        results = self._detector.detect_for_video(mp_image, timestamp_ms)

        if draw:
            _draw_crosshair(frame, cx, cy)
        
        if not results.face_landmarks:
            self._ema_landmarks = None
            self._ema_scale = None
            self._ema_angle = None
            return None, None, None
        
        # Pick the face closest to center if not locked
        target_idx = 0
        if self.is_locked:
            # Simple persistent tracking: pick the face closest to the previous nose position or just index 0
            # For this MVP, we'll stick to index 0 for simplicity or closest to center if multiple
            target_idx = 0
            min_dist = float('inf')
            for i, face in enumerate(results.face_landmarks):
                nose = face[1]
                dist = abs(nose.x - 0.5)
                if dist < min_dist:
                    min_dist = dist
                    target_idx = i
        
        target_face = results.face_landmarks[target_idx]
        
        # Smooth landmarks
        if self._ema_landmarks is None:
            self._ema_landmarks = [[l.x, l.y, l.z] for l in target_face]
        else:
            for i in range(len(target_face)):
                self._ema_landmarks[i][0] = self._alpha * target_face[i].x + (1 - self._alpha) * self._ema_landmarks[i][0]
                self._ema_landmarks[i][1] = self._alpha * target_face[i].y + (1 - self._alpha) * self._ema_landmarks[i][1]
                self._ema_landmarks[i][2] = self._alpha * target_face[i].z + (1 - self._alpha) * self._ema_landmarks[i][2]

        # Use smoothed landmarks for everything
        smoothed_face = [type('obj', (object,), {'x': l[0], 'y': l[1], 'z': l[2]}) for l in self._ema_landmarks]
        
        nose_x = smoothed_face[1].x
        target_angle = _angle_from_nose_x(nose_x)
        
        # Generate stable face crop
        face_crop = self._align_face(frame, smoothed_face)

        if draw:
            # Draw standard landmarks on the original frame
            for i, face in enumerate(results.face_landmarks):
                color = COLOR_TARGET if i == target_idx else COLOR_BG_FACE
                # Just draw a few key points (eyes, nose, mouth)
                for pt_idx in [33, 263, 1, 61, 291]:
                    nx, ny = int(face[pt_idx].x * w), int(face[pt_idx].y * h)
                    cv2.circle(frame, (nx, ny), 3 if i == target_idx else 1, color, -1, cv2.LINE_AA)
            
            if self.is_locked:
                sign = "+" if target_angle >= 0 else ""
                label = f"LOCKED: {sign}{target_angle:.1f}°"
                # Draw label top left
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TARGET, 2, cv2.LINE_AA)

        return target_angle, None, face_crop
