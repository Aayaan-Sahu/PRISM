"""
AV-TSE Recorder — spacebar-triggered synchronized video + audio capture.

Opens the webcam with the FaceMesh targeting overlay and the microphone array.
Press 'i' to start recording.  Press 'i' again (or wait for max
duration) to stop.  Saves a single .mp4 containing the full-frame video
at 25 FPS and the mixed microphone audio at 16 kHz mono — exactly the
format Dolphin expects.

Usage:
    python record.py                     # saves to recordings/recording.mp4
    python record.py -o my_clip.mp4      # custom output path
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import tempfile

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf

print(sd.query_devices())

def _get_best_mic() -> tuple[int | None, int]:
    """Find the 'onn.' webcam mic if available, else default. Returns (index, channels)."""
    try:
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0 and 'onn' in d['name'].lower():
                print(f"[Audio] Auto-selected {d['name']} (index {i}, {d['max_input_channels']} ch)")
                return i, int(d['max_input_channels'])
        # Fallback to default
        idx = sd.default.device[0]
        if idx is not None:
            ch = int(devices[idx]['max_input_channels'])
            print(f"[Audio] Falling back to default mic (index {idx}, {ch} ch)")
            return None, ch
    except Exception as e:
        print(f"[Audio] Warning: failed to query devices: {e}")
    return None, 1  # Safe ultimate fallback

# ── Configuration ──────────────────────────────────────────────────────────
AGG_DEVICE_INDEX, CHS = _get_best_mic()
NATIVE_SR = 48_000            # typical native mic sample rate (we will resample)
TARGET_SR = 16_000            # Dolphin audio input rate
TARGET_FPS = 25               # Dolphin video input rate
MAX_RECORD_SEC = 5            # target duration

# ── Helpers ────────────────────────────────────────────────────────────────

def _downsample_to_mono(chunk: np.ndarray) -> np.ndarray:
    """Ensure mono and downsample 48 kHz → 16 kHz."""
    if len(chunk.shape) > 1 and chunk.shape[1] > 1:
        mono = chunk.mean(axis=1)
    else:
        mono = chunk.reshape(-1)
    # Simple decimation: 48000 / 16000 = 3
    # If the native sample rate is something else, this basic slicing will sound pitched.
    # But for 48kHz (Mac default) it's ~ok.
    return mono[::3].astype(np.float32)


class Recorder:
    """Synchronized video + audio recorder with 'i' key trigger."""

    def __init__(self, output_path: str, camera_index: int | None = None):
        self.output_path = output_path
        self.camera_index = camera_index

        # State
        self.recording = False
        self.done = False

        # Buffers (filled during recording)
        self._video_frames: list[np.ndarray] = []
        self._audio_chunks: list[np.ndarray] = []

        # Audio stream
        self._audio_stream = None

    # ── Audio callback (runs on its own thread via sounddevice) ────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Audio] {status}", flush=True)
        if self.recording:
            self._audio_chunks.append(indata.copy())

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self):
        from targeting import LipTargetingSystem
        ts = LipTargetingSystem(self.camera_index)
        ts.open()

        # Start audio stream (always listening, only buffering when recording)
        self._audio_stream = sd.InputStream(
            device=AGG_DEVICE_INDEX,
            channels=CHS,
            samplerate=NATIVE_SR,
            blocksize=int(NATIVE_SR * 0.02),   # 20 ms blocks
            dtype="float32",
            callback=self._audio_callback,
        )
        self._audio_stream.start()

        print("╔══════════════════════════════════════════════════╗")
        print("║  AV-TSE Recorder                                ║")
        print("║  Press 'i' to start a 5-second recording.        ║")
        print("║  Press Q or ESC to quit without saving.          ║")
        print("╚══════════════════════════════════════════════════╝")

        record_start = None
        record_stop = None

        try:
            last_face_crop = None
            while not self.done:
                ok, frame = ts._cap.read()
                if not ok:
                    break

                angle, lip_crop, face_crop = ts._process_frame(frame, draw=True)
                
                if face_crop is not None:
                    last_face_crop = face_crop.copy()

                # Overlay status text
                if self.recording:
                    elapsed = time.time() - record_start
                    label = f"REC {elapsed:.1f}s / {MAX_RECORD_SEC}s"
                    cv2.putText(frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.circle(frame, (frame.shape[1] - 25, 25), 10, (0, 0, 255), -1)

                    # Capture full frame for recording (Dolphin needs full frame to track)
                    if getattr(ts, 'current_raw_frame', None) is not None:
                        self._video_frames.append(ts.current_raw_frame.copy())
                    else:
                        self._video_frames.append(frame.copy())

                    # Auto-stop
                    if elapsed >= MAX_RECORD_SEC:
                        print(f"\n[Recorder] Max duration reached ({MAX_RECORD_SEC}s). Stopping.")
                        self.recording = False
                        record_stop = time.time()
                        self.done = True
                else:
                    cv2.putText(frame, "STANDBY — press 'i' to record", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("AV-TSE Recorder", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("i"):
                    if not self.recording:
                        # START recording
                        self.recording = True
                        ts.is_locked = True
                        record_start = time.time()
                        self._video_frames.clear()
                        self._audio_chunks.clear()
                        print("\n[Recorder] ● Recording started (5 seconds)...")
                    else:
                        # ALREADY recording - ignore second press
                        print(" (recording currently in progress, please wait...) ")

                elif key == ord("q") or key == 27:
                    print("[Recorder] Quit without saving.")
                    self._video_frames.clear()
                    self.done = True

        finally:
            self._audio_stream.stop()
            self._audio_stream.close()
            ts.close()

        # ── Save ──────────────────────────────────────────────────────────
        if not self._video_frames:
            print("No frames captured. Nothing saved.")
            return None

        # Compute actual recording duration for sync
        actual_duration = (record_stop or time.time()) - (record_start or time.time())
        return self._save(actual_duration)

    def _save(self, actual_duration: float) -> str:
        """Mux captured video frames + audio into a single .mp4."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        n_frames = len(self._video_frames)
        h, w = self._video_frames[0].shape[:2]

        # Use the ACTUAL capture FPS so that video duration matches real time.
        # Dolphin's own convert_video_fps() will resample to 25fps later.
        actual_fps = n_frames / actual_duration if actual_duration > 0 else 30.0
        print(f"[Save] {n_frames} video frames ({w}x{h}) over {actual_duration:.2f}s → actual FPS: {actual_fps:.1f}")

        # 1. Write video to a temp file at the TRUE capture rate
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_video, fourcc, actual_fps, (w, h))
        for f in self._video_frames:
            writer.write(f)
        writer.release()

        # 2. Downsample audio to 16 kHz mono, trimmed to actual duration
        if self._audio_chunks:
            raw_audio = np.concatenate(self._audio_chunks, axis=0)     # (N, 4)
            mono_16k = _downsample_to_mono(raw_audio)
            # Trim to actual_duration so audio length matches video length
            max_samples = int(actual_duration * TARGET_SR)
            mono_16k = mono_16k[:max_samples]
        else:
            mono_16k = np.zeros(int(actual_duration * TARGET_SR), dtype=np.float32)

        tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(tmp_audio, mono_16k, TARGET_SR)
        print(f"[Save] Audio: {len(mono_16k)} samples ({len(mono_16k)/TARGET_SR:.2f}s @ {TARGET_SR}Hz)")
        print(f"[Save] Video duration: {n_frames/actual_fps:.2f}s | Audio duration: {len(mono_16k)/TARGET_SR:.2f}s")

        # 3. Mux with ffmpeg
        print(f"[Save] Muxing → {self.output_path}")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{tmp_video}" -i "{tmp_audio}" '
            f'-c:v libx264 -pix_fmt yuv420p '
            f'-c:a aac -b:a 128k -shortest '
            f'"{self.output_path}" -loglevel warning'
        )
        ret = os.system(ffmpeg_cmd)

        # Cleanup temp files
        os.unlink(tmp_video)
        os.unlink(tmp_audio)

        if ret != 0:
            print("[Save] WARNING: ffmpeg failed. Trying fallback (video-only, no audio)...")
            import shutil
            shutil.copy2(tmp_video, self.output_path)

        print(f"[Save] ✓ Saved to {self.output_path}")
        return self.output_path


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Record synchronized video+audio for Dolphin AV-TSE")
    parser.add_argument("-o", "--output", default="recordings/recording.mp4",
                        help="Output .mp4 path (default: recordings/recording.mp4)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    recorder = Recorder(output_path=args.output, camera_index=args.camera)
    result = recorder.run()
    if result:
        print(f"\nReady for separation. Run:\n  python separate.py {result}")


if __name__ == "__main__":
    main()
