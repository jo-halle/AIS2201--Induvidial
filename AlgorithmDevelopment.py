import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, lfilter
from scipy.io import wavfile

# Helper Functions
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

def slidingDFT(signal, fs, N, step):
    freqs = []
    for start in range(0, len(signal) - N, step):
        segment = signal[start:start + N]
        freqs.append(baselineFrequencyDetection(segment, fs, N))
    return freqs

def bandpassFilter(signal, fs, lowcut, highcut, order=5):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def evaluateAccuracy(detectedFreq, trueFreq):
    return np.abs(detectedFreq - trueFreq) / trueFreq * 100

def plotMagnitudeSpectrumComparison(signals, fs, labels, title, freq_cap=6000):
    plt.figure()
    for signal, label in zip(signals, labels):
        N = len(signal)
        dft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, 1 / fs)
        positiveFreqs = freqs[:N // 2]
        positiveMagnitude = np.abs(dft[:N // 2])

        # Cap the frequency range
        cap_idx = positiveFreqs <= freq_cap
        cappedFreqs = positiveFreqs[cap_idx]
        cappedMagnitude = positiveMagnitude[cap_idx]

        # Use a dotted line for specific labels
        linestyle = 'dotted' if "Zero-Padded" in label or "Filtered" in label else 'solid'
        plt.plot(cappedFreqs, cappedMagnitude, linestyle=linestyle, label=label)

    plt.title(f"Magnitude Spectrum Comparison - {title}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()
    plt.show()

# Main Test Framework
if __name__ == "__main__":
    fs = 8000
    duration = 2
    N = 1024
    step = 128

    # Test 1: Noisy Sine Wave
    t, noisySine = sineWithNoise(440, fs, duration, 0.1)
    baseline_freq = baselineFrequencyDetection(noisySine, fs, N)
    zero_padded_sine = np.pad(noisySine[:N], (0, 3 * N))  # Zero-padding
    zero_padded_freq = baselineFrequencyDetection(zero_padded_sine, fs, 4 * N)
    print(f"Noisy Sine Wave - Baseline Frequency: {baseline_freq:.2f} Hz, Error: {evaluateAccuracy(baseline_freq, 440):.2f}%")
    print(f"Noisy Sine Wave - Zero-Padded Frequency: {zero_padded_freq:.2f} Hz, Error: {evaluateAccuracy(zero_padded_freq, 440):.2f}%")
    plotMagnitudeSpectrumComparison(
        signals=[noisySine[:N], zero_padded_sine],
        fs=fs,
        labels=["Original", "Zero-Padded"],
        title="Noisy Sine Wave"
    )

    # Test 2: Abrupt Frequency Change
    t, abruptChangeSignal = abruptFreqChange(440, 880, fs, duration)
    sliding_freqs = slidingDFT(abruptChangeSignal, fs, N, step)
    # print(f"Sliding DFT Detected Frequencies: {sliding_freqs}")
    plt.figure()
    plt.plot(np.arange(len(sliding_freqs)) * step / fs, sliding_freqs, marker='o')
    plt.title("Sliding DFT for Abrupt Frequency Change")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.grid()
    plt.show()

    # Test 3: Musical Signal
    musicalFilePath = "./SampleAudio/B_oboe.wav"
    fsMusical, musicalSignal = loadWavFile(musicalFilePath)
    baseline_freq = baselineFrequencyDetection(musicalSignal, fsMusical, N)
    filtered_signal = bandpassFilter(musicalSignal, fsMusical, 100, 4200)
    filtered_freq = baselineFrequencyDetection(filtered_signal, fsMusical, N)
    print(f"Musical Signal - Baseline Frequency: {baseline_freq:.2f} Hz, Error: {evaluateAccuracy(baseline_freq, 440):.2f}%")
    print(f"Musical Signal - Filtered Frequency: {filtered_freq:.2f} Hz, Error: {evaluateAccuracy(filtered_freq, 440):.2f}%")
    plotMagnitudeSpectrumComparison(
        signals=[musicalSignal, filtered_signal],
        fs=fsMusical,
        labels=["Original", "Filtered"],
        title="Musical Signal"
    )

    # Test 4: Vocal Signal
    vocalFilePath = "./SampleAudio/Zauberflöte_vocal.wav"
    fsVocal, vocalSignal = loadWavFile(vocalFilePath)
    baseline_freq = baselineFrequencyDetection(vocalSignal, fsVocal, N)
    filtered_signal = bandpassFilter(vocalSignal, fsVocal, 100, 4200)
    filtered_freq = baselineFrequencyDetection(filtered_signal, fsVocal, N)
    print(f"Vocal Signal - Baseline Frequency: {baseline_freq:.2f} Hz, Error: {evaluateAccuracy(baseline_freq, 440):.2f}%")
    print(f"Vocal Signal - Filtered Frequency: {filtered_freq:.2f} Hz, Error: {evaluateAccuracy(filtered_freq, 440):.2f}%")
    plotMagnitudeSpectrumComparison(
        signals=[vocalSignal, filtered_signal],
        fs=fsVocal,
        labels=["Original", "Filtered"],
        title="Vocal Signal"
    )
