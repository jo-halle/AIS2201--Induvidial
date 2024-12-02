import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Functions from Part 1 (Baseline)
def loadWavFile(filePath):
    fs, data = wavfile.read(filePath)
    data = data / (2**15) if data.dtype == 'int16' else data
    return fs, data

def sineWithNoise(freq, fs, duration, noiseLevel):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sineWave = np.sin(2 * np.pi * freq * t)
    noise = noiseLevel * np.random.normal(size=t.shape)
    return t, sineWave + noise

def abruptFreqChange(freq1, freq2, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    half = len(t) // 2
    wave = np.sin(2 * np.pi * freq1 * t[:half])
    wave = np.concatenate([wave, np.sin(2 * np.pi * freq2 * t[half:])])
    return t, wave

def baselineFrequencyDetection(signal, fs, N):
    dft = np.fft.fft(signal[:N])
    freqs = np.fft.fftfreq(N, 1 / fs)
    dominantFreq = freqs[np.argmax(np.abs(dft))]
    return abs(dominantFreq)

# Solutions from Part 2
# 1. Zero-padding for Improved Frequency Resolution
def frequencyDetectionWithZeroPadding(signal, fs, N, padding_factor=2):
    padded_signal = np.pad(signal[:N], (0, N * (padding_factor - 1)), mode='constant')
    dft = np.fft.fft(padded_signal)
    freqs = np.fft.fftfreq(len(padded_signal), 1 / fs)
    dominant_freq = freqs[np.argmax(np.abs(dft))]
    return abs(dominant_freq)

# 2. Low-pass Filtering
def lowPassFilter(signal, fs, cutoff=1000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

def frequencyDetectionWithFiltering(signal, fs, N, cutoff=1000):
    filtered_signal = lowPassFilter(signal, fs, cutoff)
    return baselineFrequencyDetection(filtered_signal, fs, N)

# 3. Harmonic Product Spectrum (HPS)
def harmonicProductSpectrum(signal, fs, N, harmonics=3):
    dft = np.fft.fft(signal[:N])
    spectrum = np.abs(dft[:N // 2])
    hps = spectrum.copy()
    for h in range(2, harmonics + 1):
        downsampled = spectrum[::h]
        hps[:len(downsampled)] *= downsampled
    freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2]
    fundamental_freq = freqs[np.argmax(hps)]
    return abs(fundamental_freq)

# Test Framework Integration
if __name__ == "__main__":
    # Sampling frequency and duration
    fs = 8000
    duration = 2
    N = 1024

    # Test: Sine wave with noise
    t, noisy_sine = sineWithNoise(freq=440, fs=fs, duration=duration, noiseLevel=0.1)

    # Baseline Detection
    detected_freq_baseline = baselineFrequencyDetection(noisy_sine, fs, N)
    print(f"Baseline Detected Frequency: {detected_freq_baseline:.2f} Hz")

    # Zero-padding Detection
    detected_freq_zp = frequencyDetectionWithZeroPadding(noisy_sine, fs, N, padding_factor=4)
    print(f"Zero-padding Detected Frequency: {detected_freq_zp:.2f} Hz")

    # Filtering Detection
    detected_freq_filtered = frequencyDetectionWithFiltering(noisy_sine, fs, N, cutoff=500)
    print(f"Filtering Detected Frequency: {detected_freq_filtered:.2f} Hz")

    # Harmonic Product Spectrum Detection
    detected_freq_hps = harmonicProductSpectrum(noisy_sine, fs, N, harmonics=5)
    print(f"HPS Detected Frequency: {detected_freq_hps:.2f} Hz")
