import sys
import json
import os
import cv2
import queue
import threading
import numpy as np
import sounddevice as sd
import websocket  # pip install websocket-client
from fast_recorder import FastRecorder
from understanding_modal_dolphin_deployment import main as run_dolphin
from understanding_enroll import enroll_from_wav, upload_embedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TODO: Set this after `modal deploy understanding_tse.py`
TSE_WS_URL = "wss://spacial-audio-recognition--spatial-audio-tse-tseruntime-serve.modal.run/ws"
TSE_SAMPLE_RATE = 16_000
TSE_CHUNK_DURATION = 0.30  # 500 ms
TSE_CHUNK_SAMPLES = int(TSE_SAMPLE_RATE * TSE_CHUNK_DURATION)  # 2400 samples

# This is a queue that holds the file paths of all the video files that are yet
# to have dolphin inference run on them
dolphin_inference_queue = queue.Queue()

# Reference to the persistent TSE WebSocket connection.
# Set by tse_streaming_thread when it connects.
# modal_worker uses this to send reload_embeddings after a new enrollment.
tse_ws = None


# ---------------------------------------------------------------------------
# Dolphin + Enrollment Worker
# ---------------------------------------------------------------------------
def modal_worker():
    # this function handles running dolphin, generating embeddings, and telling modal
    # to hot reload the embeddings so they work on the fly
    while True:
        video_filepath = dolphin_inference_queue.get()
        if video_filepath is None:
            break

        print(f"[WORKER] Popped {video_filepath} from queue. Starting inference...")
        try:
            clean_wav = run_dolphin(video_filepath)  # returns path to Dolphin-cleaned .wav

            if clean_wav is None:
                print("[WORKER] Dolphin returned no audio, skipping enrollment")
                continue

            # 1. Enroll locally: saves embedding0.npy + embedding0_ref.npy in embeddings/
            out_path = enroll_from_wav(clean_wav)
            ref_path = out_path.replace(".npy", "_ref.npy")

            # 2. Upload both files to the Modal Volume so the running TSE container
            #    can see them. Both must be uploaded — TSE skips speakers missing _ref.npy
            upload_embedding(
                emb_filename=os.path.basename(out_path),
                emb_data=open(out_path, "rb").read(),
                ref_filename=os.path.basename(ref_path),
                ref_data=open(ref_path, "rb").read(),
            )
            print(f"[WORKER] Uploaded {os.path.basename(out_path)} to Modal Volume")

            # 3. Tell the running TSE to reload from the updated Volume
            if tse_ws is not None:
                try:
                    tse_ws.send(json.dumps({"type": "reload_embeddings"}))
                    print("[WORKER] Sent reload_embeddings to TSE")
                except Exception as ws_err:
                    print(f"[WORKER] Could not notify TSE: {ws_err}")

        except Exception as e:
            print(f"[WORKER] Error processing {video_filepath}: {e}")
        finally:
            dolphin_inference_queue.task_done()


def queue_dolphin_inference(filepath):
    print(
        f"[ORCHESTRATOR] Recording saved to {filepath}. Adding to processing queue..."
    )
    dolphin_inference_queue.put(filepath)


# ---------------------------------------------------------------------------
# TSE Streaming Thread
# ---------------------------------------------------------------------------
def tse_streaming_thread(shutdown_event: threading.Event):
    """
    Persistent background thread that:
    1. Connects to the Modal TSE WebSocket
    2. Captures mic audio continuously (its own InputStream, always listening)
    3. Sends 150 ms PCM16 chunks to the server
    4. Receives filtered audio back and plays it through speakers

    The server performs WeSep separation + ECAPA verification for each enrolled
    speaker. Only matched speaker audio comes back; everything else is silence.
    """
    global tse_ws

    # Queue to pass mic chunks from the audio callback to the send loop.
    # Audio callbacks must never block, so we decouple via a queue.
    send_queue = queue.Queue()

    # Queue to pass filtered audio from the recv path to the playback callback.
    playback_queue = queue.Queue()

    # Detect the mic's native sample rate
    native_sr = int(sd.query_devices(None, "input")["default_samplerate"])
    needs_resample = native_sr != TSE_SAMPLE_RATE

    # Build a resampler if the mic doesn't natively run at 16 kHz.
    # Uses the same torchaudio approach as enroll_demo.py.
    resampler = None
    if needs_resample:
        import torch
        import torchaudio
        resampler = torchaudio.transforms.Resample(native_sr, TSE_SAMPLE_RATE)
        print(f"[TSE CLIENT] Resampling mic {native_sr} Hz -> {TSE_SAMPLE_RATE} Hz")

    # The mic callback runs on sounddevice's audio thread — must be fast, no blocking.
    # We just dump the raw mono audio into send_queue for the main loop to process.
    mic_block_size = int(native_sr * TSE_CHUNK_DURATION)

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[TSE CLIENT] Mic warning: {status}")
        send_queue.put(indata[:, 0].copy())  # mono float32

    # The speaker callback pulls filtered audio from the playback queue.
    # If nothing is queued, output silence so the stream stays smooth.
    def speaker_callback(outdata, frames, time_info, status):
        try:
            chunk = playback_queue.get_nowait()
            n = min(len(chunk), len(outdata))
            outdata[:n, 0] = chunk[:n]
            outdata[n:, 0] = 0.0
        except queue.Empty:
            outdata[:, 0] = 0.0

    # --- Connect to Modal TSE WebSocket ---
    print(f"[TSE CLIENT] Connecting to {TSE_WS_URL}...")
    try:
        ws = websocket.WebSocket()
        ws.connect(TSE_WS_URL)
    except Exception as e:
        print(f"[TSE CLIENT] Failed to connect: {e}")
        return

    tse_ws = ws  # expose globally so modal_worker can send reload_embeddings

    # --- Wait for the "ready" handshake ---
    try:
        ready_raw = ws.recv()
        ready_msg = json.loads(ready_raw)
        if ready_msg.get("type") != "ready":
            print(f"[TSE CLIENT] Unexpected first message: {ready_msg}")
            ws.close()
            tse_ws = None
            return
        print(
            f"[TSE CLIENT] Connected! "
            f"speakers={ready_msg['speakers_loaded']}, "
            f"sr={ready_msg['sample_rate']}, "
            f"chunk={ready_msg['chunk_samples']}"
        )
    except Exception as e:
        print(f"[TSE CLIENT] Handshake failed: {e}")
        ws.close()
        tse_ws = None
        return

    # --- Inner thread: send mic audio to server ---
    def _tse_send_loop():
        sends_count = 0
        try:
            while not shutdown_event.is_set():
                try:
                    raw_chunk = send_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Resample to 16 kHz if mic runs at a different rate
                if resampler is not None:
                    import torch
                    tensor = torch.from_numpy(raw_chunk).float().unsqueeze(0)
                    resampled = resampler(tensor).squeeze().numpy()
                else:
                    resampled = raw_chunk

                # Convert float32 [-1, 1] -> PCM16 and send
                pcm16 = (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16)
                ws.send_binary(pcm16.tobytes())
                sends_count += 1

                # Debug: log every 20th send
                if sends_count % 20 == 0:
                    in_energy = float(np.mean(resampled ** 2))
                    print(f"[TSE SEND] #{sends_count} energy={in_energy:.6f}")

        except websocket.WebSocketConnectionClosedException:
            print("[TSE SEND] WebSocket closed")
        except Exception as e:
            print(f"[TSE SEND] Error: {e}")

    # --- Inner thread: receive filtered audio from server ---
    def _tse_recv_loop():
        recv_count = 0
        try:
            while not shutdown_event.is_set():
                try:
                    opcode, response = ws.recv_data()
                except websocket.WebSocketTimeoutException:
                    continue
                except websocket.WebSocketConnectionClosedException:
                    break

                if opcode == websocket.ABNF.OPCODE_BINARY:
                    # Filtered PCM16 audio — decode and queue for playback
                    filtered = np.frombuffer(response, dtype=np.int16).astype(np.float32) / 32768.0
                    playback_queue.put(filtered)
                    recv_count += 1

                    # Debug: log received audio energy every 20th chunk
                    if recv_count % 20 == 0:
                        out_energy = float(np.mean(filtered ** 2))
                        is_silence = out_energy < 1e-8
                        print(f"[TSE RECV] #{recv_count} energy={out_energy:.6f} {'(silence)' if is_silence else '(has audio)'}")

                elif opcode == websocket.ABNF.OPCODE_TEXT:
                    msg = json.loads(response.decode("utf-8"))
                    msg_type = msg.get("type", "")

                    if msg_type == "embeddings_loaded":
                        print(f"[TSE] Reloaded: {msg['count']} speaker(s)")
                    elif msg_type == "error":
                        print(f"[TSE] Server error: {msg['detail']}")

        except Exception as e:
            print(f"[TSE RECV] Error: {e}")

    # --- Open audio streams and run send/recv on separate threads ---
    try:
        with sd.InputStream(
            samplerate=native_sr,
            channels=1,
            blocksize=mic_block_size,
            callback=mic_callback,
        ):
            with sd.OutputStream(
                samplerate=TSE_SAMPLE_RATE,
                channels=1,
                blocksize=TSE_CHUNK_SAMPLES,
                callback=speaker_callback,
            ):
                print("[TSE CLIENT] Streaming started (mic -> TSE -> speakers)")

                send_t = threading.Thread(target=_tse_send_loop, daemon=True)
                recv_t = threading.Thread(target=_tse_recv_loop, daemon=True)
                send_t.start()
                recv_t.start()

                # Block until shutdown — keeps audio streams alive
                send_t.join()
                recv_t.join()

    except Exception as e:
        print(f"[TSE CLIENT] Streaming error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[TSE CLIENT] Shutting down...")
        try:
            ws.close()
        except Exception:
            pass
        tse_ws = None


# ---------------------------------------------------------------------------
# Main — orchestrates camera, recording, Dolphin, and TSE threads
# ---------------------------------------------------------------------------
def main(camera_id=0):
    # Shared event to coordinate shutdown of the TSE streaming thread
    shutdown_event = threading.Event()

    # Thread 1: Dolphin inference + enrollment worker
    modal_worker_thread = threading.Thread(target=modal_worker, daemon=True)
    modal_worker_thread.start()

    # Thread 2: TSE streaming (always listening to mic, always playing back)
    tse_thread = threading.Thread(
        target=tse_streaming_thread,
        args=(shutdown_event,),
        daemon=True,
    )
    tse_thread.start()

    # Start cv2 camera
    print(f"[CAMERA] Attempting to open camera with ID {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[CAMERA ERROR] Could not open camera with ID {camera_id}.")
        shutdown_event.set()
        return
    print(f"[CAMERA] Successfully opened camera {camera_id}.")

    # FastRecorder: when recording stops, the finished video is queued for Dolphin
    recorder = FastRecorder(on_record_complete=queue_dolphin_inference)

    print("\n" + "=" * 50)
    print("SPATIAL AUDIO — FAST RECORDER + TSE")
    print("Press 'i' to toggle recording (enroll a speaker).")
    print("Press 'q' or ESC to quit.")
    print("TSE is always listening in the background.")
    print("=" * 50 + "\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[CAMERA ERROR] Failed to read frame.")
            break

        # Pass frame to recorder immediately
        # process_frame owns the saving video file logic
        recorder.process_frame(frame)

        # Draw recording indicator if active
        display_frame = frame.copy()
        if recorder.is_recording:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                display_frame,
                "REC",
                (50, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Fast Recorder", display_frame)

        # handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("i"):
            recorder.is_recording = not recorder.is_recording
            if recorder.is_recording:
                print("\n[RECORDER] Recording STARTED.")
            else:
                print("\n[RECORDER] Recording STOPPED.")
        elif key == ord("q") or key == 27:
            break

    # Clean up
    print("\n[ORCHESTRATOR] Shutting down...")

    # Signal TSE thread to stop
    shutdown_event.set()
    tse_thread.join(timeout=3)

    recorder.close()
    cap.release()
    cv2.destroyAllWindows()

    print("[ORCHESTRATOR] Waiting for pending cloud inferences to finish...")
    dolphin_inference_queue.join()

    # Send poison pill to stop the worker thread
    dolphin_inference_queue.put(None)
    modal_worker_thread.join()
    print("[ORCHESTRATOR] All clean. Goodbye!")


if __name__ == "__main__":
    cam_id = 0
    if len(sys.argv) > 1:
        try:
            cam_id = int(sys.argv[1])
        except ValueError:
            print("[WARNING] Invalid camera ID provided. Defaulting to 0.")

    main(cam_id)
