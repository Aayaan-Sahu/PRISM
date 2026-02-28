import torch
import torchaudio
import numpy as np

# Quick test script to see if PyTorch STFT/ISTFT logic works for cleaning
def test_pytorch_stft():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    
    # Fake waveform 300ms at 48kHz
    sr = 48000
    length = int(0.3 * sr)
    waveform = torch.randn(1, length).to(device)
    
    # Fake noise spectrum size
    FFT_SIZE = 1024
    HOP_SIZE = 512
    
    noise_spectrum = torch.rand(FFT_SIZE // 2 + 1).to(device) * 0.5
    
    window = torch.hann_window(FFT_SIZE).to(device)
    
    # STFT
    stft = torch.stft(waveform, n_fft=FFT_SIZE, hop_length=HOP_SIZE, window=window, return_complex=True)
    mag = stft.abs()
    phase = stft.angle()
    
    # Subtract noise
    # mag is [1, freq, frames]
    # noise_spectrum is [freq]
    noise_expanded = noise_spectrum.unsqueeze(0).unsqueeze(2)
    
    clean_mag = torch.maximum(mag - 4.0 * noise_expanded, 0.01 * mag)
    clean_stft = clean_mag * torch.exp(1j * phase)
    
    # ISTFT
    clean_waveform = torch.istft(clean_stft, n_fft=FFT_SIZE, hop_length=HOP_SIZE, window=window, length=length)
    
    print("Clean waveform shape:", clean_waveform.shape)
    print("Success")

if __name__ == "__main__":
    test_pytorch_stft()
