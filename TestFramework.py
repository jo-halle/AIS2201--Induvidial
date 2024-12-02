import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram
from scipy.io import wavfile

# Test Signal Generators
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

def loadWavFile(filePath):
    fs, data = wavfile.read(filePath)
    data = data / (2**15) if data.dtype == 'int16' else data
    return fs, data

# Frequency Detection Functions
def baselineFrequencyDetection(signal, fs, N):
    dft = np.fft.fft(signal[:N])
    freqs = np.fft.fftfreq(N, 1 / fs)
    return abs(freqs[np.argmax(np.abs(dft))])

def calculateError(detectedFreq, trueFreq):
    return np.abs(detectedFreq - trueFreq) / trueFreq * 100

# Filtering
def bandPassFilter(signal, fs, lowCutoff, highCutoff):
    nyquist = 0.5 * fs
    low = lowCutoff / nyquist
    high = highCutoff / nyquist
    b, a = butter(5, [low, high], btype='band')
    return lfilter(b, a, signal)

# Sliding DFT
def slidingDFT(signal, fs, windowSize, stepSize):
    freqs = []
    for i in range(0, len(signal) - windowSize, stepSize):
        freqs.append(baselineFrequencyDetection(signal[i:i + windowSize], fs, windowSize))
    return freqs

# Visualization
def plotSignal(time, signal, title):
    plt.figure()
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
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

def plotSlidingDFT(timeStamps, frequencies, title):
    plt.figure()
    plt.plot(timeStamps, frequencies, marker='o')
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.show()

# Test Framework
fs = 8000  # Sampling frequency
duration = 2  # Signal duration in seconds
N = 1024  # DFT size

# 1. Sine Wave with Noise
t, noisySine = sineWithNoise(440, fs, duration, 0.1)
detectedFreqNoisy = baselineFrequencyDetection(noisySine, fs, N)
errorNoisy = calculateError(detectedFreqNoisy, 440)
print(f"Noisy Sine Wave - Detected Frequency: {detectedFreqNoisy} Hz, Error: {errorNoisy:.2f}%")
plotSignal(t, noisySine, "Sine Wave with Noise")

# 2. Abrupt Frequency Change
t, abruptChange = abruptFreqChange(440, 880, fs, duration)
detectedFreqsAbrupt = slidingDFT(abruptChange, fs, windowSize=1024, stepSize=512)
print(f"Abrupt Change Signal - Detected Frequencies: {detectedFreqsAbrupt}")
plotSlidingDFT(np.arange(len(detectedFreqsAbrupt)) * 512 / fs, detectedFreqsAbrupt, "Sliding DFT for Abrupt Frequency Change")

# 3. Musical Signal
musicalFile = "./SampleAudio/B_oboe.wav"  # Replace with the path to your musical WAV file
fsMusical, musicalSignal = loadWavFile(musicalFile)
filteredMusicalSignal = bandPassFilter(musicalSignal, fsMusical, 400, 500)
detectedFreqMusical = baselineFrequencyDetection(filteredMusicalSignal, fsMusical, N)
print(f"Musical Signal - Detected Frequency: {detectedFreqMusical} Hz")
plotSpectrogram(filteredMusicalSignal, fsMusical, "Filtered Musical Signal")

# 4. Vocal Signal
vocalFile = "./SampleAudio/Zauberflöte_vocal.wav"  # Replace with the path to your vocal WAV file
fsVocal, vocalSignal = loadWavFile(vocalFile)
slidingFreqsVocal = slidingDFT(vocalSignal, fsVocal, windowSize=1024, stepSize=512)
print(f"Vocal Signal - Detected Frequencies: {slidingFreqsVocal}")
plotSlidingDFT(np.arange(len(slidingFreqsVocal)) * 512 / fsVocal, slidingFreqsVocal, "Sliding DFT for Vocal Signal")
