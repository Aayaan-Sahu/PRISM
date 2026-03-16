import argparse

import os
import queue
import shutil
import threading
from understanding_modal_dolphin_deployment import main as run_dolphin
from understanding_targeting import LipTargetingSystem
from understanding_record import Recorder

# queue.Queue is thread safe
dolphin_inference_queue = queue.Queue()

def modal_worker():
    while True:
        video_filepath = dolphin_inference_queue.get()

        if video_filepath is None:
            break

        print(f"[WORKER] Popped {video_filepath} from queue. Starting inference...")
        try:
            run_dolphin(video_filepath)
        except Exception as e:
            print(f"[WORKER] Error processing {video_filepath}: {e}")
        finally:
            # if os.path.exists(video_filepath):
            #     os.remove(video_filepath)
            #     print(f"[WORKER] Cleaned up temporary input file: {video_filepath}")
            
            dolphin_inference_queue.task_done()

def queue_dolphin_inference(filepath):
    print(f"[ORCHESTRATOR] Recording saved to {filepath}. Adding to processing queue...")
    dolphin_inference_queue.put(filepath)

def main():
    modal_worker_thread = threading.Thread(target=modal_worker, daemon=True)
    modal_worker_thread.start()

    with LipTargetingSystem(camera_index=0) as targeting_system:
        av_recorder = Recorder(targeting_system=targeting_system, on_record_complete=queue_dolphin_inference)

        for angle, lip_crop, face_crop in targeting_system.stream(display=True):
            av_recorder.process_frame(angle, face_crop)

        av_recorder.close()
    
    print("\n[ORCHESTRATOR] Shutting down. Waiting for pending cloud inferences to finish...")
    dolphin_inference_queue.join() 
    
    # Send poison pill to stop the worker thread
    dolphin_inference_queue.put(None)
    modal_worker_thread.join()
    print("[ORCHESTRATOR] All clean. Goodbye!")


if __name__ == "__main__":
    main()
