import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from scipy.signal import resample_poly

# Baseline frequency detection function
def freq_detection(x_n, fs, N=1024):
    timestamps = []
    freqs = []
    for window_start in range(0, len(x_n) - N + 1, N):
        window_end = window_start + N
        x_slice = x_n[window_start:window_end]
        X_m = np.fft.rfft(x_slice, n=N)
        X_m[0] = 0
        m_peak = np.argmax(np.abs(X_m))
        freqs.append(m_peak * fs / N)
        timestamps.append(window_end / fs)
    return np.array(timestamps), np.array(freqs)

# Helper Functions
def sine_wave_with_noise(f, fs, duration, noise_level):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * f * t)
    noise = noise_level * np.random.normal(size=t.shape)
    return t, sine_wave + noise

def abrupt_frequency_change(f1, f2, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    split_point = len(t) // 2
    wave = np.concatenate([
        np.sin(2 * np.pi * f1 * t[:split_point]),
        np.sin(2 * np.pi * f2 * t[split_point:])
    ])
    return t, wave

def add_noise(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def plot_with_shaded_error(snr_values, errors, label, color):
    mean_errors = [np.mean(err) for err in errors]
    std_errors = [np.std(err) for err in errors]
    
    plt.plot(snr_values, mean_errors, label=label, color=color)
    plt.fill_between(
        snr_values,
        np.array(mean_errors) - np.array(std_errors),
        np.array(mean_errors) + np.array(std_errors),
        color=color,
        alpha=0.3
    )

# Test functions
def test_sine_wave_with_noise():
    fs = 10000
    f = 440
    duration = 2
    snr_values = np.linspace(-10, 30, 10)
    errors = []
    for snr_db in snr_values:
        _, noisy_signal = sine_wave_with_noise(f, fs, duration, 0)
        noisy_signal = add_noise(noisy_signal, snr_db)
        _, freqs = freq_detection(noisy_signal, fs, N=1024)
        errors.append(np.abs(freqs - f))
    plot_with_shaded_error(snr_values, errors, "Sine Wave + Noise", "blue")
    plt.title("Frequency Detection Error vs SNR (Sine Wave)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error (Hz)")
    plt.legend()
    plt.grid()
    plt.show()

def test_abrupt_frequency_change():
    fs = 10000
    f1, f2 = 440, 880
    duration = 2
    _, signal = abrupt_frequency_change(f1, f2, fs, duration)
    for N in [1024, 2048]:
        timestamps, freqs = freq_detection(signal, fs, N=N)
        plt.plot(timestamps, freqs, label=f"N={N}")
    plt.axhline(f1, color='blue', linestyle='--', label="True f1")
    plt.axhline(f2, color='red', linestyle='--', label="True f2")
    plt.title("Abrupt Frequency Change")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid()
    plt.show()

def test_musical_signal():
    fs, signal = wavfile.read('./SampleAudio/B_oboe.wav')
    signal = signal / np.max(np.abs(signal))
    snr_values = np.linspace(-10, 30, 10)
    errors = []
    for snr_db in snr_values:
        noisy_signal = add_noise(signal, snr_db)
        _, freqs = freq_detection(noisy_signal, fs, N=1024)
        errors.append(np.abs(freqs - 440))  # Assume A4 note
    plot_with_shaded_error(snr_values, errors, "Musical Signal + Noise", "green")
    plt.title("Frequency Detection Error vs SNR (Musical Signal)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error (Hz)")
    plt.legend()
    plt.grid()
    plt.show()

def test_vocal_signal():
    fs, signal = wavfile.read('./SampleAudio/Zauberflöte_vocal.wav')
    signal = signal / np.max(np.abs(signal))
    snr_values = np.linspace(-10, 30, 10)
    errors = []
    for snr_db in snr_values:
        noisy_signal = add_noise(signal, snr_db)
        _, freqs = freq_detection(noisy_signal, fs, N=1024)
        errors.append(np.abs(freqs - 440))  # Assume A4 note
    plot_with_shaded_error(snr_values, errors, "Vocal Signal + Noise", "red")
    plt.title("Frequency Detection Error vs SNR (Vocal Signal)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error (Hz)")
    plt.legend()
    plt.grid()
    plt.show()

# Main Execution
if __name__ == "__main__":
    test_sine_wave_with_noise()
    test_abrupt_frequency_change()
    test_musical_signal()
    test_vocal_signal()