import numpy as np
import sounddevice as sd
import wave
import time


print(sd.query_devices())

# ─── Configuration ───────────────────────────────────────────────
AGG_DEVICE_INDEX = 9          # Aggregate Device (4 in, 0 out)
CHS = 4                       # 2 channels per Yeti × 2 Yetis

MIC_DISTANCE_M = 0.4953       # measured mic spacing in metres
SPEED_OF_SOUND = 343.0        # m/s

WINDOW_SEC = 0.20             # analysis window length

CALIBRATION_SEC = 2.0         # seconds of ambient noise to calibrate
SPIKE_SIGMA = 3.0             # std-devs above mean = "spike"

# ── Target angle range ──────────────────────────────────────────
ANGLE_MIN = -60.0             # degrees
ANGLE_MAX = -30.0             # degrees

RECORD_SEC = 10               # how long to record beamformed audio

# GCC-PHAT confidence: reject peaks below this value
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
    R /= np.abs(R) + EPS

    cc = np.fft.irfft(R, n=n_fft)

    # only keep lags within the physically possible range
    max_shift = int(np.ceil(max_tau * fs)) + 1
    max_shift = min(max_shift, n_fft // 2)

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

    # confidence = peak value relative to mean of valid region
    peak_val = float(np.abs(cc_valid[peak]))
    mean_val = float(np.mean(np.abs(cc_valid))) + EPS
    confidence = peak_val / mean_val

    return tau, confidence


# ─── Delay-and-sum beamformer ────────────────────────────────────
def beamform(mic_a, mic_b, steer_angle_deg, fs):
    """
    Apply delay-and-sum beamforming steered toward steer_angle_deg.
    Returns the beamformed (mono) signal.
    """
    # steering delay in seconds
    tau_steer = MIC_DISTANCE_M * np.sin(np.radians(steer_angle_deg)) / SPEED_OF_SOUND

    n_fft = 1
    while n_fft < len(mic_a):
        n_fft <<= 1

    # apply fractional delay to mic_b via frequency-domain phase shift
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * tau_steer)

    B = np.fft.rfft(mic_b, n=n_fft)
    mic_b_delayed = np.fft.irfft(B * phase_shift, n=n_fft)[: len(mic_a)]

    # sum and normalize
    output = 0.5 * (mic_a + mic_b_delayed)
    return output.astype(np.float32)


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
    print(f"Target angle range: [{ANGLE_MIN:+.1f}°, {ANGLE_MAX:+.1f}°]")

    max_tau = MIC_DISTANCE_M / SPEED_OF_SOUND
    threshold = calibrate(fs)

    steer_angle = (ANGLE_MIN + ANGLE_MAX) / 2.0
    print(f"Beamformer steered to: {steer_angle:+.1f}°")

    win = int(fs * WINDOW_SEC)
    ring = np.zeros((win, CHS), dtype=np.float32)
    filled = 0

    # accumulate beamformed audio for saving
    output_chunks = []
    start_time = time.time()

    # hold mechanism: keep outputting beamformed audio for a short
    # period after the last confirmed in-range frame, so that brief
    # dips in energy/confidence/angle don't cut out mid-word.
    # ~15 frames × 21ms per frame ≈ 300ms hold time.
    hold_frames = 15
    hold_counter = 0

    def callback(indata, frames, time_info, status):
        nonlocal ring, filled, hold_counter

        x = np.asarray(indata[:, :CHS], dtype=np.float32)
        n = x.shape[0]
        silence = np.zeros(n, dtype=np.float32)

        if n >= win:
            ring[:] = x[-win:]
            filled = win
        else:
            ring[:-n] = ring[n:]
            ring[-n:] = x
            filled = min(win, filled + n)

        # ring buffer not full yet
        if filled < win:
            output_chunks.append(silence)
            return

        mic_a = 0.5 * (ring[:, 0] + ring[:, 1])
        mic_b = 0.5 * (ring[:, 2] + ring[:, 3])

        # spike check
        energy = 0.5 * (np.mean(mic_a ** 2) + np.mean(mic_b ** 2))

        if energy >= threshold:
            # enough energy — run GCC-PHAT
            tau, confidence = gcc_phat_tau(mic_a, mic_b, fs, max_tau)

            if confidence >= MIN_CONFIDENCE:
                sin_theta = np.clip(
                    (tau * SPEED_OF_SOUND) / MIC_DISTANCE_M, -1.0, 1.0
                )
                angle_deg = float(np.degrees(np.arcsin(sin_theta)))

                if ANGLE_MIN <= angle_deg <= ANGLE_MAX:
                    # confirmed in-range — refresh the hold counter
                    hold_counter = hold_frames
                    bf = beamform(mic_a, mic_b, steer_angle, fs)
                    output_chunks.append(bf[-n:].copy())
                    print(
                        f">>> IN RANGE | angle: {angle_deg:+.1f}° | "
                        f"confidence: {confidence:.2f}",
                        flush=True,
                    )
                    return

        # not a confirmed in-range frame — but are we still in a hold period?
        if hold_counter > 0:
            hold_counter -= 1
            # keep beamforming toward the target direction to bridge the gap
            bf = beamform(mic_a, mic_b, steer_angle, fs)
            output_chunks.append(bf[-n:].copy())
            print(f"    (hold: {hold_counter} frames left)", flush=True)
        else:
            output_chunks.append(silence)
            print("(nothing unusual)", flush=True)

    print(f"\nListening for {RECORD_SEC}s… (Ctrl+C to stop early)\n")

    try:
        with sd.InputStream(
            device=AGG_DEVICE_INDEX,
            channels=CHS,
            samplerate=fs,
            blocksize=1024,
            dtype="float32",
            callback=callback,
        ):
            while time.time() - start_time < RECORD_SEC:
                sd.sleep(100)
    except KeyboardInterrupt:
        pass

    # save beamformed output
    if output_chunks:
        output = np.concatenate(output_chunks)
        filename = "beamformed_output.wav"

        with wave.open(filename, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(fs)
            pcm = np.clip(output * 32767, -32768, 32767).astype(np.int16)
            wf.writeframes(pcm.tobytes())

        print(f"\nSaved {len(output)} samples ({len(output)/fs:.2f}s) → {filename}")
    else:
        print("\nNo audio from target direction was captured.")


if __name__ == "__main__":
    main()
