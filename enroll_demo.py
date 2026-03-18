import collections
import sys
import numpy as np
import sounddevice as sd
import torch
import torchaudio

from understanding_enroll import enroll_from_wav, _load_model, _MODEL_SR

def compute_cosine_similarity(v1, v2):
    """Computes the cosine similarity between two 1D vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def run_demo(wav_path: str):
    # Determine compute device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading embedding for {wav_path}...")
    
    # 1. Compute and save the embedding for the enrolled file, utilizing existing logic
    try:
        saved_emb_path = enroll_from_wav(wav_path, device=device)
        target_emb = np.load(saved_emb_path)
    except Exception as e:
        print(f"Error enrolling audio file: {e}")
        return

    print(f"Target embedding loaded. Shape: {target_emb.shape}")
    
    # 2. Load model for inference over mic audio
    model = _load_model(device)
    
    # 3. Setup microphone and rolling buffer
    WINDOW_DURATION_S = 2.0  # 2 seconds of context
    STEP_DURATION_S = 0.5    # Process every 0.5 seconds

    native_sr = int(sd.query_devices(None, "input")["default_samplerate"])
    
    # Calculate sizes in samples
    window_samples = int(_MODEL_SR * WINDOW_DURATION_S)
    step_samples = int(native_sr * STEP_DURATION_S)

    # Create a rolling buffer mapped to model sr
    audio_buffer = collections.deque(maxlen=window_samples)
    
    # Pre-instantiate resampler if needed
    resampler = None
    if native_sr != _MODEL_SR:
        print(f"Will resample mic input from {native_sr}Hz to {_MODEL_SR}Hz")
        resampler = torchaudio.transforms.Resample(native_sr, _MODEL_SR).to(device)

    THRESHOLD = 0.35  # Arbitrary threshold for Match / No Match determination

    print(f"\n--- Starting microphone stream ---")
    print(f"Listening in {STEP_DURATION_S}s chunks with a {WINDOW_DURATION_S}s rolling window.")
    print(f"Similarity threshold is {THRESHOLD}.")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    def audio_callback(indata, frames, time_info, status):
        # We process this STEP_DURATION_S chunk
        energy = np.mean(np.abs(indata))
        
        # Convert audio chunk from numpy arrays of shape (frames, 1 channel) for mono
        chunk_tensor = torch.from_numpy(indata.T).float().to(device)
        
        if resampler:
            chunk_tensor = resampler(chunk_tensor)
            
        # Add the newest 0.5s equivalent samples back to CPU NumPy to store in deque
        resampled_chunk = chunk_tensor.squeeze().cpu().numpy()
        
        # We append zeros if it's completely silent to avoid processing "old" voice
        # indefinitely if the user stops talking, else append the actual audio
        if energy < 0.005:
            audio_buffer.extend(np.zeros_like(resampled_chunk))
        else:
            audio_buffer.extend(resampled_chunk)
            
        # Only evaluate if we have a full window (2.0s)
        if len(audio_buffer) < window_samples:
            return
            
        # If the overall buffer has very little energy, it's silence
        buffer_array = np.array(audio_buffer)
        if np.mean(np.abs(buffer_array)) < 0.005:
            return
            
        # Process the 2-second buffer
        waveform = torch.from_numpy(buffer_array).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            emb = model.encode_batch(waveform)
            chunk_emb = emb.squeeze().cpu().numpy()
            
        similarity = compute_cosine_similarity(target_emb, chunk_emb)
        
        if similarity >= THRESHOLD:
            print(f"Match!     (score: {similarity:.3f})")
        else:
            print(f"No match   (score: {similarity:.3f})")

    try:
        # Request mono input stream using step_samples as blocksize
        with sd.InputStream(samplerate=native_sr, channels=1, blocksize=step_samples, callback=audio_callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"\nFailed to start microphone stream: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python enroll_demo.py <path_to_audio_file>")
        sys.exit(1)
    
    wav_file_path = sys.argv[1]
    run_demo(wav_file_path)
