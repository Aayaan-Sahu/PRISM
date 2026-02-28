"""
Live Embedding Matching Experiment
===================================

Interactively registers two reference speakers' voices. Continuously records
live audio, separates multiple talking streams, and checks them against
the registered speakers.
Streams matching the speakers are amplified (G_FG), others are attenuated (G_BG).
The entire processed recording is saved to one output file.
"""

import sys
import wave
import signal
import numpy as np
import sounddevice as sd
import torch
import torchaudio

from embeddings import SpeakerEncoder, verify_speaker, VERIFICATION_THRESHOLD
from separator import SpeechSeparator
from config import G_FG, G_BG, TARGET_SR

# Global flag for the capture loop
running = True

def main():
    print("Live Embedding Matching Experiment\n")

    # 1. Initialise models
    print("🔧 Loading models …")
    encoder = SpeakerEncoder()
    separator = SpeechSeparator()

    # 2. Register Reference Speakers
    duration = 5.0
    fs = TARGET_SR

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
    print(f"   Matching speakers receive {G_FG}x gain.")
    print(f"   Others receive {G_BG}x gain.\n")

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

                print("-" * 40)
                try:
                    streams = separator.separate(chunk_tensor, sr=fs)
                except Exception as e:
                    print(f"Separation failed: {e}. Attenuating as background.")
                    output_chunks.append(chunk * G_BG)
                    continue

                if not streams:
                    print("No speech detected.")
                    output_chunks.append(chunk * G_BG)
                    continue

                mixed = torch.zeros(chunk_tensor.shape[1])
                
                for i, stream_tensor in enumerate(streams):
                    # stream_tensor is 1-D: [time]
                    # Align lengths if separation produced differently sized tensors
                    if stream_tensor.shape[0] > mixed.shape[0]:
                        stream_tensor = stream_tensor[:mixed.shape[0]]
                    elif stream_tensor.shape[0] < mixed.shape[0]:
                        padded = torch.zeros_like(mixed)
                        padded[:stream_tensor.shape[0]] = stream_tensor
                        stream_tensor = padded

                    # Check match for the specific separated stream
                    # encoder expects [1, T] or just 1D tensor
                    emb_stream = encoder.encode(stream_tensor)
                    
                    same_1, score_1 = verify_speaker(emb_ref_1, emb_stream)
                    same_2, score_2 = verify_speaker(emb_ref_2, emb_stream)
                    
                    if same_1 or same_2:
                        target_gain = G_FG
                        match_str = "Speaker 1 & 2" if (same_1 and same_2) else ("Speaker 1" if same_1 else "Speaker 2")
                        print(f"✅ Stream {i}: Match! ({match_str}) (scores: {score_1:+.2f}, {score_2:+.2f}) -> Gain: {G_FG}")
                    else:
                        target_gain = G_BG
                        print(f"❌ Stream {i}: No Match (scores: {score_1:+.2f}, {score_2:+.2f}) -> Gain: {G_BG}")

                    mixed += target_gain * stream_tensor

                # Peak limiter
                peak = mixed.abs().max()
                if peak > 1.0:
                    mixed = mixed / peak

                # Convert mixed back to numpy [chunk_samples, 1] for output
                processed_chunk = mixed.unsqueeze(1).numpy()
                output_chunks.append(processed_chunk)
                    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"An error occurred during live capture: {e}")

    # 4. Save to file
    if output_chunks:
        output_audio = np.concatenate(output_chunks)
        filename = "live_embedding_matching_output.wav"
        
        with wave.open(filename, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            pcm = np.clip(output_audio * 32767, -32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())
            
        print(f"\n💾 Saved {len(output_audio)/fs:.2f}s of continuous audio to {filename}")
    else:
        print("\n🤷 No audio was processed.")

if __name__ == "__main__":
    main()
