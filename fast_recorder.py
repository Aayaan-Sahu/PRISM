import os
import cv2
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RECORDINGS_DIR = os.path.join(_REPO_ROOT, "recordings")

class FastRecorder:
    def __init__(
        self,
        model_sample_rate=16000,
        output_prefix="output_fast",
        on_record_complete=None,
    ):
        self.is_recording = False
        self.sample_rate = int(sd.query_devices(None, "input")["default_samplerate"])
        self.model_sample_rate = model_sample_rate
        self.output_prefix = output_prefix
        self.on_record_complete = on_record_complete

        # Stores tuple: (timestamp, frame)
        self.recorded_frames = []
        self.recorded_audio_chunks = []

        self.last_valid_frame = None

        self.was_recording = False
        self.recording_start_time = None
        self.recording_stop_time = None

        os.makedirs(_RECORDINGS_DIR, exist_ok=True)

        self.buffer_lock = threading.Lock()
        self.shutdown_requested = threading.Event()

        self.audio_thread = threading.Thread(
            target=self._stream_audio_task, daemon=True
        )
        self.audio_thread.start()

    def _resample_audio(
        self, audio: np.ndarray, src_sr: int, dst_sr: int
    ) -> np.ndarray:
        # _resample_audio handles changing the input audio into the corect sampling rate
        if src_sr == dst_sr:
            return audio.astype(np.float32, copy=False)

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) == 0:
            return np.zeros((0,), dtype=np.float32)

        src_times = np.arange(len(audio), dtype=np.float64) / float(src_sr)
        duration = len(audio) / float(src_sr)
        dst_len = max(1, int(round(duration * dst_sr)))
        dst_times = np.arange(dst_len, dtype=np.float64) / float(dst_sr)

        resampled = np.interp(dst_times, src_times, audio).astype(np.float32)
        return resampled

    def _prepare_model_audio(
        self, audio_data: np.ndarray, duration_seconds: float
    ) -> np.ndarray:
        audio_data = np.asarray(audio_data, dtype=np.float32)

        # ensure audio is mono
        if audio_data.ndim == 2:
            if audio_data.shape[1] == 1:
                mono = audio_data[:, 0]
            else:
                mono = audio_data.mean(axis=1)
        else:
            mono = audio_data.reshape(-1)

        # ensure that the audio is the correct sampling rate
        mono = self._resample_audio(mono, self.sample_rate, self.model_sample_rate)
        target_samples = max(1, int(round(duration_seconds * self.model_sample_rate)))

        if len(mono) < target_samples:
            mono = np.pad(mono, (0, target_samples - len(mono)))
        else:
            mono = mono[:target_samples]

        return mono.astype(np.float32, copy=False)

    def _stream_audio_task(self):
        # this function is a thread function that appends audio data as it comes in
        # the InputStream opens a new thread that calls the audio callback which appends audio data
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"[AUDIO WARNING] {status}")
            if self.is_recording:
                chunk_timestamp = time.perf_counter()
                with self.buffer_lock:
                    self.recorded_audio_chunks.append((chunk_timestamp, indata.copy()))

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, callback=audio_callback
        ):
            self.shutdown_requested.wait()
        print("[AUDIO] Mic stream closed.")

    def _merge_video_task(self, video_frames, audio_chunks, run_id):
        if not video_frames or not audio_chunks:
            return

        # the following logic handles making sure the video and audio are synced
        start_ts = max(video_frames[0][0], audio_chunks[0][0])
        end_ts = min(video_frames[-1][0], audio_chunks[-1][0])

        if end_ts <= start_ts:
            return

        video_frames = [
            (ts, frame) for ts, frame in video_frames if start_ts <= ts <= end_ts
        ]
        audio_chunks = [(ts, chunk) for ts, chunk in audio_chunks if ts <= end_ts]

        if not video_frames or not audio_chunks:
            return

        print(
            f"[BACKGROUND] Saving {len(video_frames)} frames to {self.output_prefix}_{run_id}.mp4"
        )

        unique_prefix = f"{self.output_prefix}_{run_id}"
        temp_video = f"temp_video_{unique_prefix}.mp4"
        temp_audio = f"temp_audio_{unique_prefix}.wav"
        temp_model_audio = f"temp_model_audio_{unique_prefix}.wav"
        final_output = os.path.join(_RECORDINGS_DIR, f"{unique_prefix}.mp4")

        # write current audio data to a temp file
        aligned_audio_parts = []
        for ts, chunk in audio_chunks:
            chunk_end = ts
            chunk_duration = len(chunk) / self.sample_rate
            chunk_start = chunk_end - chunk_duration

            overlap_start = max(chunk_start, start_ts)
            overlap_end = min(chunk_end, end_ts)

            if overlap_end <= overlap_start:
                continue

            start_idx = int(round((overlap_start - chunk_start) * self.sample_rate))
            end_idx = int(round((overlap_end - chunk_start) * self.sample_rate))
            start_idx = max(0, min(start_idx, len(chunk)))
            end_idx = max(start_idx, min(end_idx, len(chunk)))

            aligned_audio_parts.append(chunk[start_idx:end_idx])

        if not aligned_audio_parts:
            return

        # save unpreprocessed audio to temp_audio
        audio_data = np.concatenate(aligned_audio_parts, axis=0)
        sf.write(temp_audio, audio_data, self.sample_rate)

        # process audio and save to temp_model_audio
        duration = len(audio_data) / self.sample_rate
        model_audio = self._prepare_model_audio(audio_data, duration)
        sf.write(temp_model_audio, model_audio, self.model_sample_rate)

        # save the video file
        fps = 30.0
        output_frame_count = max(1, int(round(duration * fps)))
        frame_times = np.array([ts for ts, _ in video_frames], dtype=np.float64)
        frames_only = [frame for _, frame in video_frames]

        h, w = frames_only[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

        for i in range(output_frame_count):
            target_ts = start_ts + (i / fps)
            idx = np.searchsorted(frame_times, target_ts, side="right") - 1
            idx = max(0, min(idx, len(frames_only) - 1))
            out.write(frames_only[idx])

        out.release()

        # create the final video file
        print("[PROCESSING] Merging tracks...")
        video_clip = VideoFileClip(temp_video)
        audio_clip = AudioFileClip(temp_audio)
        final_clip = video_clip.with_audio(audio_clip)
        final_clip.write_videofile(final_output, codec="libx264", audio_codec="aac")
        final_clip.close()
        video_clip.close()
        audio_clip.close()

        if os.path.exists(temp_video):
            os.remove(temp_video)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        if os.path.exists(temp_model_audio):
            os.remove(temp_model_audio)

        print(f"[SUCCESS] Finished rendering {final_output}")

        # if callback exists, call it -> right now this runs inference on the newly saved video file
        if self.on_record_complete:
            self.on_record_complete(final_output)

    def process_frame(self, raw_frame):
        now = time.perf_counter()

        # make sure no empty frames
        if raw_frame is not None:
            self.last_valid_frame = raw_frame.copy()

        # this tracks when we START recording
        if not self.was_recording and self.is_recording:
            self.recording_start_time = now
            self.recording_stop_time = None

        # this tracks when we STOP recording
        if self.was_recording and not self.is_recording:
            self.recording_stop_time = now

            # use mutex lock
            with self.buffer_lock:
                frames_copy = self.recorded_frames.copy()
                audio_copy = self.recorded_audio_chunks.copy()
                self.recorded_frames.clear()
                self.recorded_audio_chunks.clear()

            # call thread that will asynchronously save the video file
            run_id = int(time.time())
            save_thread = threading.Thread(
                target=self._merge_video_task, args=(frames_copy, audio_copy, run_id)
            )
            save_thread.start()

        # append the current frame to recorded frames
        if self.is_recording:
            frame_to_append = (
                raw_frame.copy()
                if raw_frame is not None
                else self.last_valid_frame.copy()
                if self.last_valid_frame is not None
                else np.zeros((720, 1280, 3), dtype=np.uint8)
            )
            with self.buffer_lock:
                self.recorded_frames.append((now, frame_to_append))

        self.was_recording = self.is_recording

    def close(self):
        self.shutdown_requested.set()
        self.audio_thread.join()
