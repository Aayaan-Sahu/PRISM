import numpy as np
import sounddevice as sd

# ─── Configuration ───────────────────────────────────────────────
AGG_DEVICE_INDEX = 9          # Aggregate Device (4 in, 0 out)
CHS = 4                       # 2 channels per Yeti × 2 Yetis

MIC_DISTANCE_M = 0.4953       # measured mic spacing in metres
SPEED_OF_SOUND = 343.0        # m/s

WINDOW_SEC = 0.20             # analysis window length

CALIBRATION_SEC = 2.0         # seconds of ambient noise to calibrate
SPIKE_SIGMA = 3.0             # std-devs above mean = "spike"

# GCC-PHAT confidence: reject peaks below this normalised value
MIN_CONFIDENCE = 1.5

EPS = 1e-12


# ─── GCC-PHAT time delay estimation (constrained) ───────────────
def gcc_phat_tau(x, y, fs, max_tau):
    """
    GCC-PHAT with search constrained to ±max_tau seconds.
    Returns (tau_seconds, confidence).
    """
    x = x - np.mean(x)
    y = y - np.mean(y)

    n_fft = 1
    while n_fft < len(x) + len(y):
        n_fft <<= 1

    X = np.fft.rfft(x, n=n_fft)
    Y = np.fft.rfft(y, n=n_fft)

    R = X * np.conj(Y)
    mag = np.abs(R)
    R /= mag + EPS

    cc = np.fft.irfft(R, n=n_fft)

    # only keep lags within the physically possible range
    max_shift = int(np.ceil(max_tau * fs)) + 1
    max_shift = min(max_shift, n_fft // 2)

    # extract valid region: negative lags from tail, positive from head
    cc_valid = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    peak = int(np.argmax(np.abs(cc_valid)))
    shift = peak - max_shift

    # sub-sample parabolic interpolation
    if 1 <= peak < len(cc_valid) - 1:
        a, b, c = cc_valid[peak - 1], cc_valid[peak], cc_valid[peak + 1]
        denom = a - 2 * b + c
        if abs(denom) > EPS:
            shift += 0.5 * (a - c) / denom

    tau = float(shift) / float(fs)

    # confidence = peak value relative to mean of the valid region
    peak_val = float(np.abs(cc_valid[peak]))
    mean_val = float(np.mean(np.abs(cc_valid))) + EPS
    confidence = peak_val / mean_val

    return tau, confidence


# ─── Calibrate ambient noise ────────────────────────────────────
def calibrate(fs):
    """Record ambient audio and return the spike threshold."""
    cal_samples = int(fs * CALIBRATION_SEC)
    print(f"Calibrating ({CALIBRATION_SEC}s — stay quiet)…")

    cal_data = sd.rec(
        cal_samples,
        samplerate=fs,
        channels=CHS,
        device=AGG_DEVICE_INDEX,
        dtype="float32",
    )
    sd.wait()

    win = int(fs * WINDOW_SEC)
    energies = []
    for start in range(0, cal_samples - win, win):
        chunk = cal_data[start : start + win]
        mic_a = 0.5 * (chunk[:, 0] + chunk[:, 1])
        mic_b = 0.5 * (chunk[:, 2] + chunk[:, 3])
        e = 0.5 * (np.mean(mic_a ** 2) + np.mean(mic_b ** 2))
        energies.append(e)

    energies = np.array(energies)
    mean_e = float(np.mean(energies))
    std_e = float(np.std(energies))
    threshold = mean_e + SPIKE_SIGMA * std_e

    print(f"  mean energy : {mean_e:.2e}")
    print(f"  std energy  : {std_e:.2e}")
    print(f"  threshold   : {threshold:.2e}  ({SPIKE_SIGMA}σ above mean)")
    return threshold


# ─── Main ────────────────────────────────────────────────────────
def main():
    info = sd.query_devices(AGG_DEVICE_INDEX, "input")
    fs = int(info["default_samplerate"])

    if int(info["max_input_channels"]) < CHS:
        raise RuntimeError(
            f"Aggregate device has only {info['max_input_channels']} "
            f"input channels, expected {CHS}."
        )

    print(f"Using Aggregate Device (index {AGG_DEVICE_INDEX}) @ {fs} Hz")

    max_tau = MIC_DISTANCE_M / SPEED_OF_SOUND
    threshold = calibrate(fs)

    win = int(fs * WINDOW_SEC)
    ring = np.zeros((win, CHS), dtype=np.float32)
    filled = 0

    def callback(indata, frames, time_info, status):
        nonlocal ring, filled

        x = np.asarray(indata[:, :CHS], dtype=np.float32)
        n = x.shape[0]

        if n >= win:
            ring[:] = x[-win:]
            filled = win
        else:
            ring[:-n] = ring[n:]
            ring[-n:] = x
            filled = min(win, filled + n)

        if filled < win:
            return

        # downmix to mono per mic (no bandpass — full bandwidth for GCC-PHAT)
        mic_a = 0.5 * (ring[:, 0] + ring[:, 1])
        mic_b = 0.5 * (ring[:, 2] + ring[:, 3])

        # spike check
        energy = 0.5 * (np.mean(mic_a ** 2) + np.mean(mic_b ** 2))
        if energy < threshold:
            print("(nothing unusual)", flush=True)
            return

        # estimate delay, constrained to physical limits
        tau, confidence = gcc_phat_tau(mic_a, mic_b, fs, max_tau)

        if confidence < MIN_CONFIDENCE:
            print(f"(low confidence: {confidence:.2f})", flush=True)
            return

        # convert delay → angle directly (no smoothing)
        sin_theta = np.clip(
            (tau * SPEED_OF_SOUND) / MIC_DISTANCE_M, -1.0, 1.0
        )
        angle_deg = float(np.degrees(np.arcsin(sin_theta)))

        print(f">>> angle: {angle_deg:+.1f}°  (confidence: {confidence:.2f})", flush=True)

    print(f"Max delay: ±{max_tau*1e6:.0f} µs  (±{int(np.ceil(max_tau * fs))} samples)")
    print("Listening… (Ctrl+C to stop)\n")

    with sd.InputStream(
        device=AGG_DEVICE_INDEX,
        channels=CHS,
        samplerate=fs,
        blocksize=1024,
        dtype="float32",
        callback=callback,
    ):
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    main()
