"""
Recorder — synchronized audio/video capture for a fixed duration.

Captures raw BGR frames and 16 kHz mono audio simultaneously, then
exports as an MP4+WAV pair muxed via ffmpeg.
"""

from __future__ import annotations

import glob as _glob
import os
import shutil
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd


_AUDIO_RATE = 16_000   # 16 kHz mono
_VIDEO_FPS = 25        # target output FPS
_RECORD_DURATION = 5.0  # seconds


@dataclass
class RecordedClip:
    """Container for a finished recording."""
    frames: list[np.ndarray]           # BGR frames at capture rate
    frame_timestamps: list[float]      # wall-clock timestamps per frame
    detections: list                    # MediaPipe detections per frame
    audio: np.ndarray                  # (N,) float32 mono @ 16 kHz
    audio_rate: int = _AUDIO_RATE
    duration: float = 0.0


class AVRecorder:
    """
    Synchronized audio/video recorder.

    Usage::

        rec = AVRecorder()
        rec.start()
        while not rec.is_done:
            rec.push_frame(frame, detection)
        clip = rec.finish()
    """

    def __init__(self, duration: float = _RECORD_DURATION) -> None:
        self._duration = duration
        self._recording = False
        self._start_time: float = 0.0

        # Video buffers
        self._frames: list[np.ndarray] = []
        self._frame_timestamps: list[float] = []
        self._detections: list = []

        # Audio
        self._audio_chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_done(self) -> bool:
        if not self._recording:
            return False
        return (time.monotonic() - self._start_time) >= self._duration

    @property
    def elapsed(self) -> float:
        if not self._recording:
            return 0.0
        return time.monotonic() - self._start_time

    def start(self) -> None:
        """Begin recording audio and accepting video frames."""
        self._frames.clear()
        self._frame_timestamps.clear()
        self._detections.clear()
        self._audio_chunks.clear()

        self._start_time = time.monotonic()
        self._recording = True

        # Start audio capture in a background thread via sounddevice
        self._stream = sd.InputStream(
            samplerate=_AUDIO_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice callback — runs in audio thread."""
        self._audio_chunks.append(indata[:, 0].copy())

    def push_frame(self, frame: np.ndarray, detection=None) -> None:
        """Store a video frame with its detection (call from capture loop)."""
        if not self._recording:
            return
        self._frames.append(frame.copy())
        self._frame_timestamps.append(time.monotonic() - self._start_time)
        self._detections.append(detection)

    def finish(self) -> RecordedClip:
        """Stop recording and return the captured clip."""
        self._recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Concatenate audio
        if self._audio_chunks:
            audio = np.concatenate(self._audio_chunks)
        else:
            audio = np.zeros(0, dtype=np.float32)

        duration = self._frame_timestamps[-1] if self._frame_timestamps else 0.0

        # Resample video to exactly 25 FPS via nearest-timestamp selection
        frames_25, dets_25 = self._resample_to_fps(
            self._frames, self._frame_timestamps, self._detections, _VIDEO_FPS, duration,
        )

        return RecordedClip(
            frames=frames_25,
            frame_timestamps=[i / _VIDEO_FPS for i in range(len(frames_25))],
            detections=dets_25,
            audio=audio,
            audio_rate=_AUDIO_RATE,
            duration=duration,
        )

    def cancel(self) -> None:
        """Abort recording without output."""
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._frames.clear()
        self._frame_timestamps.clear()
        self._detections.clear()
        self._audio_chunks.clear()

    @staticmethod
    def _resample_to_fps(
        frames: list[np.ndarray],
        timestamps: list[float],
        detections: list,
        target_fps: int,
        duration: float,
    ) -> tuple[list[np.ndarray], list]:
        """Resample frames to target FPS using nearest-timestamp selection."""
        if not frames:
            return [], []

        n_out = max(1, int(duration * target_fps))
        ts_arr = np.array(timestamps)
        out_frames = []
        out_dets = []

        for i in range(n_out):
            t = i / target_fps
            idx = int(np.argmin(np.abs(ts_arr - t)))
            out_frames.append(frames[idx])
            out_dets.append(detections[idx])

        return out_frames, out_dets


def save_clip(clip: RecordedClip, output_dir: str) -> tuple[str, str]:
    """
    Save a RecordedClip as video.mp4 + audio.wav in output_dir.

    Uses cv2.VideoWriter for raw video, wave module for audio,
    then muxes with ffmpeg.

    Returns (video_path, audio_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "video.mp4")
    audio_path = os.path.join(output_dir, "audio.wav")

    # Write audio WAV
    _write_wav(audio_path, clip.audio, clip.audio_rate)

    if not clip.frames:
        return video_path, audio_path

    h, w = clip.frames[0].shape[:2]

    # Try ffmpeg mux first; fall back to direct OpenCV write
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(tmp_path, fourcc, _VIDEO_FPS, (w, h))
            for frame in clip.frames:
                writer.write(frame)
            writer.release()

            cmd = [
                ffmpeg, "-y",
                "-i", tmp_path,
                "-i", audio_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                video_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Direct write — video only (audio stays as separate .wav)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, _VIDEO_FPS, (w, h))
        for frame in clip.frames:
            writer.write(frame)
        writer.release()

    return video_path, audio_path


def _find_ffmpeg() -> str | None:
    """Find the ffmpeg executable — checks PATH then common install locations."""
    # Check PATH first
    path = shutil.which("ffmpeg")
    if path:
        return path

    # Check WinGet install location
    winget_pattern = os.path.expanduser(
        "~/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"
    )
    matches = _glob.glob(winget_pattern)
    if matches:
        return matches[0]

    return None


def _write_wav(path: str, audio: np.ndarray, rate: int) -> None:
    """Write float32 audio array to a 16-bit PCM WAV file."""
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
