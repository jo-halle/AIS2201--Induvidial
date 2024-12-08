import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from FreqDetection import CombinedFreqDetection

Fs = 10_000
NDFT = 1024

def generateFreqStepSignal(Fs, tTotal=2.0, f1=440, f2=880, amplitude=1, noiseVar=0.5):
    nTotal = int(tTotal * Fs)
    timeN = np.arange(nTotal) / Fs
    timeChange = tTotal / 2
    nChange = int(timeChange * Fs)

    signal = np.zeros(nTotal)
    signal[:nChange] = amplitude * np.sin(2 * np.pi * f1 * timeN[:nChange])
    signal[nChange:] = amplitude * np.sin(2 * np.pi * f2 * timeN[nChange:])
    
    noise = np.random.normal(scale=np.sqrt(noiseVar), size=nTotal)
    xN = signal + noise
    trueFreqs = np.where(timeN < timeChange, f1, f2)
    return timeN, xN, trueFreqs

def generatePureSineNoise(Fs, duration=4.0, frequency=885, amplitude=1.0, noiseVar=0.1):
    n = int(duration * Fs)
    timeN = np.arange(n) / Fs
    cleanSignal = amplitude * np.sin(2 * np.pi * frequency * timeN)
    noise = np.random.normal(scale=np.sqrt(noiseVar), size=n)
    noisySignal = cleanSignal + noise
    return timeN, noisySignal, frequency

def loadAndPrepareAudio(audioFile, targetFs=16000, duration=2.2, trueFrequency=440):
    origFs, audioData = wavfile.read(audioFile)
    audioData = audioData.astype(np.float32) / np.max(np.abs(audioData))
    n = int(duration * targetFs)

    if origFs != targetFs:
        audioData = resample_poly(audioData, targetFs, origFs)

    if len(audioData) < n:
        audioData = np.pad(audioData, (0, n - len(audioData)), 'constant')
    else:
        audioData = audioData[:n]

    signalPower = np.mean(audioData**2)
    return audioData, signalPower, n, trueFrequency, targetFs

def TestCombinedFreqDetection():
    # Test 1: Frequency Step Signal
    TimeN_step, XN_step, TrueFreqs_step = generateFreqStepSignal(Fs)
    T_combined_step, F_combined_step = CombinedFreqDetection(XN_step, Fs, N=1024, padFactor=4, harmonics=3)
    Error_step = np.abs(F_combined_step - TrueFreqs_step[:len(F_combined_step)])

    # Test 2: Pure Sine with Noise
    TimeN_sine, XN_sine, TrueFreq_sine = generatePureSineNoise(Fs)
    T_combined_sine, F_combined_sine = CombinedFreqDetection(XN_sine, Fs, N=1024, padFactor=4, harmonics=3)
    Error_sine = np.abs(F_combined_sine - TrueFreq_sine)

    # Test 3: Real-world Audio (A4 Oboe)
    audioFile1 = './SampleAudio/A4_oboe.wav'
    AudioData1, _, _, TrueFreq1, Fs1 = loadAndPrepareAudio(audioFile1, targetFs=Fs)
    T_combined_audio1, F_combined_audio1 = CombinedFreqDetection(AudioData1, Fs1, N=1024, padFactor=4, harmonics=3)
    Error_audio1 = np.abs(F_combined_audio1 - TrueFreq1)

    # Test 4: Real-world Audio (Vocal)
    audioFile2 = './SampleAudio/Zauberflöte_vocal.wav'
    AudioData2, _, _, TrueFreq2, Fs2 = loadAndPrepareAudio(audioFile2, targetFs=Fs)
    T_combined_audio2, F_combined_audio2 = CombinedFreqDetection(AudioData2, Fs2, N=1024, padFactor=4, harmonics=3)
    Error_audio2 = np.abs(F_combined_audio2 - TrueFreq2)

    # Plot results
    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Combined Frequency Detection Algorithm - Frequency Estimate and Error")

    # Frequency Step Signal
    axs[0, 0].plot(TimeN_step, TrueFreqs_step, 'b', label='True Frequency')
    axs[0, 0].plot(T_combined_step, F_combined_step, 'r', label='Detected Frequency')
    axs[0, 0].fill_between(T_combined_step, F_combined_step - Error_step, F_combined_step + Error_step, color='r', alpha=0.2)
    axs[0, 0].set_title("Frequency Step Signal - Frequency Estimate")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Frequency (Hz)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(T_combined_step, Error_step, 'r')
    axs[0, 1].set_title("Frequency Step Signal - Error")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Error (Hz)")
    axs[0, 1].grid(True)

    # Pure Sine with Noise
    axs[1, 0].axhline(TrueFreq_sine, color='b', linestyle='--', label=f'True Frequency = {TrueFreq_sine} Hz')
    axs[1, 0].plot(T_combined_sine, F_combined_sine, 'r', label='Detected Frequency')
    axs[1, 0].fill_between(T_combined_sine, F_combined_sine - Error_sine, F_combined_sine + Error_sine, color='r', alpha=0.2)
    axs[1, 0].set_title("Pure Sine with Noise - Frequency Estimate")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(T_combined_sine, Error_sine, 'r')
    axs[1, 1].set_title("Pure Sine with Noise - Error")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Error (Hz)")
    axs[1, 1].grid(True)

    # Real-world Audio (A4 Oboe)
    axs[2, 0].axhline(TrueFreq1, color='b', linestyle='--', label=f'True Frequency = {TrueFreq1} Hz')
    axs[2, 0].plot(T_combined_audio1, F_combined_audio1, 'r', label='Detected Frequency')
    axs[2, 0].fill_between(T_combined_audio1, F_combined_audio1 - Error_audio1, F_combined_audio1 + Error_audio1, color='r', alpha=0.2)
    axs[2, 0].set_title("Real-world Audio (A4 Oboe) - Frequency Estimate")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Frequency (Hz)")
    axs[2, 0].grid(True)
    axs[2, 0].legend()

    axs[2, 1].plot(T_combined_audio1, Error_audio1, 'r')
    axs[2, 1].set_title("Real-world Audio (A4 Oboe) - Error")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Error (Hz)")
    axs[2, 1].grid(True)

    # Real-world Audio (Vocal)
    axs[3, 0].axhline(TrueFreq2, color='b', linestyle='--', label=f'True Frequency = {TrueFreq2} Hz')
    axs[3, 0].plot(T_combined_audio2, F_combined_audio2, 'r', label='Detected Frequency')
    axs[3, 0].fill_between(T_combined_audio2, F_combined_audio2 - Error_audio2, F_combined_audio2 + Error_audio2, color='r', alpha=0.2)
    axs[3, 0].set_title("Real-world Audio (Vocal) - Frequency Estimate")
    axs[3, 0].set_xlabel("Time (s)")
    axs[3, 0].set_ylabel("Frequency (Hz)")
    axs[3, 0].grid(True)
    axs[3, 0].legend()

    axs[3, 1].plot(T_combined_audio2, Error_audio2, 'r')
    axs[3, 1].set_title("Real-world Audio (Vocal) - Error")
    axs[3, 1].set_xlabel("Time (s)")
    axs[3, 1].set_ylabel("Error (Hz)")
    axs[3, 1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TestCombinedFreqDetection()
