import numpy as np
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

# === PARAMETERS ===
FILENAME = 'ppg_data.txt'
FS = 50  # Sampling frequency in Hz (adjust based on your real sensor)

# === STEP 1: Read PPG data ===
def read_ppg_data(filename):
    with open(filename, 'r') as file:
        data = file.read().replace('\n', ',')
    data = [float(val.strip()) for val in data.split(',') if val.strip()]
    return np.array(data)

# === STEP 2: Bandpass filter to remove noise ===
def bandpass_filter(data, fs, lowcut=0.1, highcut=2.0):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, data)

# === STEP 3: Extract envelope using Hilbert transform ===
def extract_envelope(ppg_signal):
    analytic_signal = hilbert(ppg_signal)
    envelope = np.abs(analytic_signal)
    return envelope

# === STEP 4: Estimate respiratory rate using FFT ===
def estimate_rr(envelope, fs):
    n = len(envelope)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    spectrum = np.abs(np.fft.rfft(envelope))

    rr_band = (freqs >= 0.1) & (freqs <= 0.5)

    # Debugging: print frequency info
    print("Min freq:", freqs.min(), "Max freq:", freqs.max())
    print("RR band count:", np.sum(rr_band))

    if not np.any(rr_band):
        raise ValueError("No frequencies found in respiratory range (0.1-0.5 Hz). Try longer or better-quality data.")

    peak_freq = freqs[rr_band][np.argmax(spectrum[rr_band])]
    rr_bpm = peak_freq * 60  # breaths per minute
    return rr_bpm

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        ppg_signal = read_ppg_data(FILENAME)
        filtered_signal = bandpass_filter(ppg_signal, FS)
        envelope = extract_envelope(filtered_signal)
        respiratory_rate = estimate_rr(envelope, FS)

        print(f"\n✅ Estimated Respiratory Rate: {respiratory_rate:.2f} breaths per minute")

        # Optional: Plot
        plt.figure(figsize=(12, 6))
        plt.plot(ppg_signal, label="Original PPG")
        plt.plot(envelope, label="Envelope (Respiration)", linestyle='--')
        plt.title("PPG Signal with Respiratory Envelope")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print("❌ Error:", e)
