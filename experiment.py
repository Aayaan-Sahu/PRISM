"""
Experiment: Selective Speaker Recording
=======================================

Interactively registers two reference speakers' voices, then continuously
records live audio. Only chunks matching either registered speaker are
saved to the final output file.
"""

import sys
import wave
import signal
import numpy as np
import sounddevice as sd
import torch

from embeddings import SpeakerEncoder, verify_speaker, VERIFICATION_THRESHOLD

# Global flag for the capture loop
running = True

def main():
    print("Selective Speaker Recording Experiment\n")

    # 1. Initialise encoder
    encoder = SpeakerEncoder()

    # 2. Register Reference Speakers
    duration = 5.0
    fs = 16000

    def record_speaker(name: str) -> torch.Tensor:
        input(f"\n🎙  Press Enter to start recording {name} for {duration} seconds...")
        print(f"🔴 Recording {name}...")
        try:
            # Using system default microphone (mono, 16kHz)
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            print("✅ Recording complete.")
            return torch.from_numpy(audio).T  # [1, T] sequence
        except Exception as e:
            print(f"Failed to record audio: {e}")
            sys.exit(1)

    audio_ref_1 = record_speaker("Reference Speaker 1")
    print(f"🎙  Encoding Reference Speaker 1 …")
    emb_ref_1 = encoder.encode(audio_ref_1)

    audio_ref_2 = record_speaker("Reference Speaker 2")
    print(f"🎙  Encoding Reference Speaker 2 …")
    emb_ref_2 = encoder.encode(audio_ref_2)
    
    # 3. Live recording loop
    chunk_sec = 2.0
    chunk_samples = int(chunk_sec * fs)

    output_chunks = []
    
    global running

    def on_sigint(sig, frame):
        global running
        running = False
        print("\n🛑 Stopping live capture...")

    signal.signal(signal.SIGINT, on_sigint)

    print("\n🎧 Listening live... (Press Ctrl+C to stop)")
    print("   Only audio matching either reference speaker will be saved.\n")

    try:
        # Open an InputStream and read chunks in a loop
        with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
            while running:
                chunk, overflowed = stream.read(chunk_samples)
                if not running:
                    break
                
                if overflowed:
                    print("⚠️ Audio buffer overflowed. Some samples may be lost.")
                
                # Check match
                # chunk is [chunk_samples, 1], so reshape to [1, T] 
                chunk_tensor = torch.from_numpy(chunk).T 
                emb_chunk = encoder.encode(chunk_tensor)
                
                same_1, score_1 = verify_speaker(emb_ref_1, emb_chunk)
                same_2, score_2 = verify_speaker(emb_ref_2, emb_chunk)
                
                if same_1 and same_2:
                    print(f"✅ Match! (Speaker 1 & 2) (scores: {score_1:+.2f}, {score_2:+.2f}) -> Saving chunk")
                    output_chunks.append(chunk)
                elif same_1:
                    print(f"✅ Match! (Speaker 1) (score: {score_1:+.2f}) -> Saving chunk")
                    output_chunks.append(chunk)
                elif same_2:
                    print(f"✅ Match! (Speaker 2) (score: {score_2:+.2f}) -> Saving chunk")
                    output_chunks.append(chunk)
                else:
                    print(f"❌ No Match (scores: {score_1:+.2f}, {score_2:+.2f}) -> Discarding chunk")
                    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred during live capture: {e}")

    # 4. Save to file
    if output_chunks:
        output_audio = np.concatenate(output_chunks)
        filename = "experiment_output.wav"
        
        with wave.open(filename, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            pcm = np.clip(output_audio * 32767, -32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
            
        print(f"\n💾 Saved {len(output_audio)/fs:.2f}s of matching audio to {filename}")
    else:
        print("\n🤷 No matching audio was captured.")


if __name__ == "__main__":
    main()
