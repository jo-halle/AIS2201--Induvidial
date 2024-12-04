import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import wavfile

# Sine wave with noise
def sine_wave_with_noise(f, fs, duration, noise_level):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * f * t)
    noise = noise_level * np.random.normal(size=t.shape)
    return t, sine_wave + noise

# Add noise to a signal
def add_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# Baseline Frequency Detection
def freq_detection(x_n, fs, N=1024):
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n) - N + 1, N):
        window_end = window_start + N
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0  # Ignore DC component
        m_peak = np.argmax(np.abs(X_m))
        freqs.append(m_peak * fs / N)
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)

# Modified Frequency Detection with Zero Padding
def freq_detection_with_padding(x_n, fs, N=1024, padding_factor=4):
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n) - N + 1, N):
        window_end = window_start + N
        x_slice = x_n[window_start:window_end]
        padded_slice = np.pad(x_slice, (0, padding_factor * N - N))
        X_m = np.fft.rfft(padded_slice)
        X_m[0] = 0
        m_peak = np.argmax(np.abs(X_m))
        freqs.append(m_peak * fs / (padding_factor * N))
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)

# Bandpass Filtering
def bandpass_filter(signal, fs, lowcut=25, highcut=4200, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

# Harmonic Averaging for Fundamental Frequency Detection
def harmonic_averaging(x_n, fs, N=1024, num_harmonics=3):
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n) - N + 1, N):
        window_end = window_start + N
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice)
        X_m[0] = 0
        harmonic_freqs = []
        for h in range(1, num_harmonics + 1):
            peak_idx = np.argmax(np.abs(X_m[h::h]))
            harmonic_freqs.append(peak_idx * fs / N / h)
        fundamental_freq = np.mean(harmonic_freqs)
        freqs.append(fundamental_freq)
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)

# Test Framework with Modifications
def test_with_modifications():
    fs = 10000
    f = 440
    duration = 2
    snr_values = np.linspace(-10, 30, 10)

    # Baseline Test
    _, noisy_signal = sine_wave_with_noise(f, fs, duration, 0)
    errors_baseline = []
    for snr_db in snr_values:
        noisy_signal = add_noise(noisy_signal, snr_db)
        _, freqs = freq_detection(noisy_signal, fs, N=1024)
        avg_error = np.mean(np.abs(freqs - f))
        errors_baseline.append(avg_error)

    # Zero Padding Test
    errors_padding = []
    for snr_db in snr_values:
        noisy_signal = add_noise(noisy_signal, snr_db)
        _, freqs = freq_detection_with_padding(noisy_signal, fs, N=1024, padding_factor=4)
        avg_error = np.mean(np.abs(freqs - f))
        errors_padding.append(avg_error)

    # Bandpass Filtering Test
    errors_filtering = []
    for snr_db in snr_values:
        noisy_signal = add_noise(noisy_signal, snr_db)
        filtered_signal = bandpass_filter(noisy_signal, fs)
        _, freqs = freq_detection(filtered_signal, fs, N=1024)
        avg_error = np.mean(np.abs(freqs - f))
        errors_filtering.append(avg_error)

    # Harmonic Averaging Test
    errors_harmonic = []
    for snr_db in snr_values:
        noisy_signal = add_noise(noisy_signal, snr_db)
        _, freqs = harmonic_averaging(noisy_signal, fs, N=1024, num_harmonics=3)
        avg_error = np.mean(np.abs(freqs - f))
        errors_harmonic.append(avg_error)

    # Plot Results
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, errors_baseline, label="Baseline Algorithm", color='blue')
    plt.plot(snr_values, errors_padding, label="Zero Padding", color='red')
    plt.plot(snr_values, errors_filtering, label="Bandpass Filtering", color='green')
    plt.plot(snr_values, errors_harmonic, label="Harmonic Averaging", color='purple')
    plt.fill_between(snr_values, np.array(errors_baseline) - np.std(errors_baseline),
                     np.array(errors_baseline) + np.std(errors_baseline), alpha=0.2, color='blue')
    plt.fill_between(snr_values, np.array(errors_padding) - np.std(errors_padding),
                     np.array(errors_padding) + np.std(errors_padding), alpha=0.2, color='red')
    plt.fill_between(snr_values, np.array(errors_filtering) - np.std(errors_filtering),
                     np.array(errors_filtering) + np.std(errors_filtering), alpha=0.2, color='green')
    plt.fill_between(snr_values, np.array(errors_harmonic) - np.std(errors_harmonic),
                     np.array(errors_harmonic) + np.std(errors_harmonic), alpha=0.2, color='purple')

    plt.title("Frequency Detection Error vs SNR (Modified Algorithms)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error (Hz)")
    plt.legend()
    plt.grid()
    plt.show()

# Main Execution
if __name__ == "__main__":
    test_with_modifications()
