import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

AGG_DEVICE_INDEX = 9
CHS = 4
WINDOW_SEC = 0.5          # how many seconds of audio to show

info = sd.query_devices(AGG_DEVICE_INDEX, "input")
fs = int(info["default_samplerate"])
win_samples = int(fs * WINDOW_SEC)

# ring buffer for each channel
buf = np.zeros((win_samples, CHS), dtype=np.float32)

def audio_callback(indata, frames, time, status):
    global buf
    x = np.asarray(indata[:, :CHS], dtype=np.float32)
    n = x.shape[0]
    buf[:-n] = buf[n:]
    buf[-n:] = x

# --- matplotlib setup ---
fig, axes = plt.subplots(CHS, 1, figsize=(12, 7), sharex=True)
fig.suptitle("Aggregate Device – Live Waveforms", fontsize=14, fontweight="bold")
fig.patch.set_facecolor("#1e1e2e")

colors = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8"]
labels = ["Yeti 1 – L", "Yeti 1 – R", "Yeti 2 – L", "Yeti 2 – R"]
t = np.linspace(0, WINDOW_SEC, win_samples)

lines = []
for i, ax in enumerate(axes):
    ax.set_facecolor("#1e1e2e")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel(labels[i], color=colors[i], fontsize=10)
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_color("#45475a")
    (line,) = ax.plot(t, np.zeros(win_samples), color=colors[i], linewidth=0.6)
    lines.append(line)

axes[-1].set_xlabel("Time (s)", color="#cdd6f4", fontsize=11)

def update(frame):
    for i, line in enumerate(lines):
        line.set_ydata(buf[:, i])
    return lines

stream = sd.InputStream(
    device=AGG_DEVICE_INDEX,
    channels=CHS,
    samplerate=fs,
    blocksize=1024,
    dtype="float32",
    callback=audio_callback,
)

print(f"Streaming from Aggregate Device (index {AGG_DEVICE_INDEX}) @ {fs} Hz")
print("Close the plot window to stop.")

with stream:
    ani = animation.FuncAnimation(fig, update, interval=30, blit=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
