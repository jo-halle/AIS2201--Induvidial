import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.io import wavfile

# Function to load WAV files
def loadWavFile(filePath):
    fs, data = wavfile.read(filePath)
    data = data / (2**15) if data.dtype == 'int16' else data
    return fs, data

# Function to generate a sine wave with noise
def sineWithNoise(freq, fs, duration, noiseLevel):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sineWave = np.sin(2 * np.pi * freq * t)
    noise = noiseLevel * np.random.normal(size=t.shape)
    return t, sineWave + noise

# Function to generate a sine wave with abrupt frequency change
def abruptFreqChange(freq1, freq2, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    half = len(t) // 2
    wave = np.sin(2 * np.pi * freq1 * t[:half])
    wave = np.concatenate([wave, np.sin(2 * np.pi * freq2 * t[half:])])
    return t, wave

# Baseline frequency detection using N-point DFT
def baselineFrequencyDetection(signal, fs, N):
    dft = np.fft.fft(signal[:N])
    freqs = np.fft.fftfreq(N, 1 / fs)
    dominantFreq = freqs[np.argmax(np.abs(dft))]
    return abs(dominantFreq)

# Function to evaluate detection accuracy
def evaluateAccuracy(detectedFreq, trueFreq):
    return np.abs(detectedFreq - trueFreq) / trueFreq * 100  # Percent error

# Spectrogram plot function
def plotSpectrogram(signal, fs, title):
    f, t, Sxx = spectrogram(signal, fs)
    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.title(f"Spectrogram - {title}")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Power [dB]")
    plt.show()

# Magnitude spectrum plot function
def plotMagnitudeSpectrum(signal, fs, title):
    N = len(signal)
    dft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Take only the positive half of the spectrum
    positive_freqs = freqs[:N // 2]
    positive_magnitude = np.abs(dft[:N // 2])
    
    plt.figure()
    plt.plot(positive_freqs, positive_magnitude)
    plt.title(f"Magnitude Spectrum - {title}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()

# Main Test Framework for Part 1
if __name__ == "__main__":
    # Sampling frequency and duration
    fs = 8000  # Sampling frequency
    duration = 2  # 2 seconds
    N = 1024  # DFT size

    # Test 1: Pure sine wave with noise
    t, noisySine = sineWithNoise(freq=440, fs=fs, duration=duration, noiseLevel=0.1)
    detectedFreq = baselineFrequencyDetection(noisySine, fs, N)
    error = evaluateAccuracy(detectedFreq, 440)
    print(f"Noisy Sine Wave - Detected Frequency: {detectedFreq:.2f} Hz, Error: {error:.2f}%")
    plotSpectrogram(noisySine, fs, "Sine Wave with Noise")
    plotMagnitudeSpectrum(noisySine, fs, "Sine Wave with Noise")

    # Test 2: Sine wave with abrupt frequency change
    t, abruptChangeSignal = abruptFreqChange(freq1=440, freq2=880, fs=fs, duration=duration)
    detectedFreqAbrupt = baselineFrequencyDetection(abruptChangeSignal, fs, N)
    print(f"Abrupt Change Signal - Detected Frequency: {detectedFreqAbrupt:.2f} Hz")
    plotSpectrogram(abruptChangeSignal, fs, "Abrupt Frequency Change Signal")
    plotMagnitudeSpectrum(abruptChangeSignal, fs, "Abrupt Frequency Change Signal")

    # Test 3: Musical instrument signal from WAV file
    musicalFilePath = "./SampleAudio/B_oboe.wav"
    fsMusical, musicalSignal = loadWavFile(musicalFilePath)
    detectedFreqMusical = baselineFrequencyDetection(musicalSignal, fsMusical, N)
    print(f"Musical Signal - Detected Frequency: {detectedFreqMusical:.2f} Hz")
    plotSpectrogram(musicalSignal, fsMusical, "Musical Signal")
    plotMagnitudeSpectrum(musicalSignal, fsMusical, "Musical Signal")

    # Test 4: Vocal-like signal from WAV file
    vocalFilePath = "./SampleAudio/Zauberflöte_vocal.wav"
    fsVocal, vocalSignal = loadWavFile(vocalFilePath)
    detectedFreqVocal = baselineFrequencyDetection(vocalSignal, fsVocal, N)
    print(f"Vocal Signal - Detected Frequency: {detectedFreqVocal:.2f} Hz")
    plotSpectrogram(vocalSignal, fsVocal, "Vocal-Like Signal")
    plotMagnitudeSpectrum(vocalSignal, fsVocal, "Vocal-Like Signal")
