import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram
from scipy.io import wavfile

# Function to load WAV files
def loadWavFile(filePath):
    fs, data = wavfile.read(filePath)
    # Normalize if data is int16
    data = data / (2**15) if data.dtype == 'int16' else data
    return fs, data

# Function to generate sine wave with noise
def sineWithNoise(freq, fs, duration, noiseLevel):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sineWave = np.sin(2 * np.pi * freq * t)
    noise = noiseLevel * np.random.normal(size=t.shape)
    return t, sineWave + noise

# Function to generate sine wave with abrupt frequency change
def abruptFreqChange(freq1, freq2, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    half = len(t) // 2
    wave = np.sin(2 * np.pi * freq1 * t[:half])
    wave = np.concatenate([wave, np.sin(2 * np.pi * freq2 * t[half:])])
    return t, wave

# Baseline frequency detection function
def baselineFrequencyDetection(signal, fs, N):
    dft = np.fft.fft(signal[:N])
    freqs = np.fft.fftfreq(N, 1 / fs)
    dominantFreq = freqs[np.argmax(np.abs(dft))]
    return abs(dominantFreq)

# Function to evaluate detection accuracy
def evaluateAccuracy(detectedFreq, trueFreq):
    return np.abs(detectedFreq - trueFreq) / trueFreq * 100  # Percent error

# Visualization helper functions
def plotSignal(time, signal, title, xlabel="Time [s]", ylabel="Amplitude"):
    plt.figure()
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plotSpectrogram(signal, fs, title):
    f, t, Sxx = spectrogram(signal, fs)
    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.title(f"Spectrogram - {title}")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="Power [dB]")
    plt.show()
    

# Sampling frequency and duration
fs = 8000  # Sampling frequency
duration = 2  # 2 seconds
N = 1024  # DFT size

# Test 1: Sine wave with noise
t, noisySine = sineWithNoise(freq=440, fs=fs, duration=duration, noiseLevel=0.1)
detectedFreq = baselineFrequencyDetection(noisySine, fs, N)
print(f"Detected frequency for noisy sine wave: {detectedFreq:.2f} Hz")
plotSignal(t, noisySine, "Sine Wave with Noise")

# Test 2: Abrupt frequency change
t, abruptChangeSignal = abruptFreqChange(freq1=440, freq2=880, fs=fs, duration=duration)
detectedFreqAbrupt = baselineFrequencyDetection(abruptChangeSignal, fs, N)
print(f"Detected frequency for abrupt change signal: {detectedFreqAbrupt:.2f} Hz")
plotSignal(t, abruptChangeSignal, "Sine Wave with Abrupt Frequency Change")

# Test 3: Musical signal from WAV file
musicalFilePath = "./SampleAudio/B_oboe.wav"
fsMusical, musicalSignal = loadWavFile(musicalFilePath)
detectedFreqMusical = baselineFrequencyDetection(musicalSignal, fsMusical, N)
print(f"Detected frequency for musical signal: {detectedFreqMusical:.2f} Hz")
plotSpectrogram(musicalSignal, fsMusical, "Musical Signal")

# Test 4: Vocal-like signal from WAV file
vocalFilePath = "./SampleAudio/Zauberflöte_vocal.wav"
fsVocal, vocalSignal = loadWavFile(vocalFilePath)
detectedFreqVocal = baselineFrequencyDetection(vocalSignal, fsVocal, N)
print(f"Detected frequency for vocal-like signal: {detectedFreqVocal:.2f} Hz")
plotSpectrogram(vocalSignal, fsVocal, "Vocal-Like Signal")
