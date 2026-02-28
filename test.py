# import sounddevice as sd

# print(sd.query_devices())

# device_index = 9  # replace with your aggregate device index

# info = sd.query_devices(device_index)
# print(info)

import sounddevice as sd
import numpy as np

fs = 48000
duration = 2

device_index = 9  # your aggregate device

print("Recording... clap near mic 1 only")
data = sd.rec(int(duration * fs),
              samplerate=fs,
              channels=4,
              device=device_index)
sd.wait()

print("Shape:", data.shape)