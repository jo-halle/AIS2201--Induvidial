import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from TestFramework import (
    sineWithNoise,
    abruptFreqChange,
    loadWavFile,
    plotSpectrogram,
    plotMagnitudeSpectrum,
    baselineFrequencyDetection,
)

# Function for zero-padding
# Function for zero-padding
def zeroPaddedDFT(signal, fs, N):
    if len(signal) > N:
        # If the signal is already longer than N, truncate it
        truncated_signal = signal[:N]
        dft = np.fft.fft(truncated_signal)
    else:
        # Otherwise, zero-pad the signal
        padded_signal = np.pad(signal, (0, N - len(signal)), 'constant')
        dft = np.fft.fft(padded_signal)
    freqs = np.fft.fftfreq(N, 1 / fs)
    dominantFreq = freqs[np.argmax(np.abs(dft))]
    return abs(dominantFreq)


# Function for band-pass filtering
def bandPassFilter(signal, fs, lowcut, highcut):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Function for sliding DFT
def slidingDFT(signal, fs, N, step_size):
    freqs = np.fft.fftfreq(N, 1 / fs)
    detected_frequencies = []
    for start in range(0, len(signal) - N + 1, step_size):
        window = signal[start:start + N]
        dft = np.fft.fft(window)
        dominantFreq = freqs[np.argmax(np.abs(dft))]
        detected_frequencies.append(abs(dominantFreq))
    return detected_frequencies

# Main test framework for Part 2
if __name__ == "__main__":
    fs = 8000  # Sampling frequency
    duration = 2  # Duration of signals
    N = 1024  # DFT size

    # Test 1: Pure sine wave with noise
    t, noisySine = sineWithNoise(freq=440, fs=fs, duration=duration, noiseLevel=0.1)
    detectedFreq = baselineFrequencyDetection(noisySine, fs, N)
    zero_padded_freq = zeroPaddedDFT(noisySine, fs, N * 4)  # 4x zero-padding
    print(f"Noisy Sine Wave - Detected Frequency: {detectedFreq:.2f} Hz")
    print(f"Noisy Sine Wave - Zero-Padded Frequency: {zero_padded_freq:.2f} Hz")
    print(f"Baseline Frequency: {detectedFreq:.2f} Hz")

    # Test 2: Sine wave with abrupt frequency change
    t, abruptSignal = abruptFreqChange(freq1=440, freq2=880, fs=fs, duration=duration)
    sliding_freqs = slidingDFT(abruptSignal, fs, N, step_size=512)
    print("Test 2: Abrupt Frequency Change")
    print(f"Sliding DFT Detected Frequencies: {sliding_freqs}")
    plt.figure()
    plt.plot(np.arange(len(sliding_freqs)) * (512 / fs), sliding_freqs, marker="o")
    plt.title("Sliding DFT for Abrupt Frequency Change")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.grid()
    plt.show()

    # Test 3: Musical signal from WAV file
    musicalFilePath = "./SampleAudio/B_oboe.wav"
    fsMusical, musicalSignal = loadWavFile(musicalFilePath)
    baseline_freq = baselineFrequencyDetection(musicalSignal, fsMusical, N)
    zero_padded_freq = zeroPaddedDFT(musicalSignal, fsMusical, N * 4)  # 4x zero-padding
    filtered_signal = bandPassFilter(musicalSignal, fsMusical, 200, 1000)
    filtered_freq = baselineFrequencyDetection(filtered_signal, fsMusical, N)

    print("Test 3: Musical Signal")
    print(f"Baseline Frequency: {baseline_freq:.2f} Hz")
    print(f"Zero-Padded Frequency: {zero_padded_freq:.2f} Hz")
    print(f"Filtered Frequency: {filtered_freq:.2f} Hz")
    plotMagnitudeSpectrum(musicalSignal, fsMusical, "Baseline - Musical Signal")
    plotMagnitudeSpectrum(filtered_signal, fsMusical, "Filtered - Musical Signal")

    # Test 4: Vocal signal from WAV file
    vocalFilePath = "./SampleAudio/Zauberflöte_vocal.wav"
    fsVocal, vocalSignal = loadWavFile(vocalFilePath)
    baseline_freq = baselineFrequencyDetection(vocalSignal, fsVocal, N)
    zero_padded_freq = zeroPaddedDFT(vocalSignal, fsVocal, N * 4)  # 4x zero-padding
    filtered_signal = bandPassFilter(vocalSignal, fsVocal, 80, 300)
    filtered_freq = baselineFrequencyDetection(filtered_signal, fsVocal, N)

    print("Test 4: Vocal Signal")
    print(f"Baseline Frequency: {baseline_freq:.2f} Hz")
    print(f"Zero-Padded Frequency: {zero_padded_freq:.2f} Hz")
    print(f"Filtered Frequency: {filtered_freq:.2f} Hz")
    plotMagnitudeSpectrum(vocalSignal, fsVocal, "Baseline - Vocal Signal")
    plotMagnitudeSpectrum(filtered_signal, fsVocal, "Filtered - Vocal Signal")
