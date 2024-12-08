import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from baseline_algorithm import freq_detection
from FreqDetection import FreqDetectionZeroPad, FreqDetectionWindowed, FreqDetectionHps, CombinedFreqDetection

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

def generatePureSineNoise(Fs, duration=4.0, frequency=885, amplitude=1.0):
    n = int(duration * Fs)
    timeN = np.arange(n) / Fs
    cleanSignal = amplitude * np.sin(2 * np.pi * frequency * timeN)
    return timeN, cleanSignal, frequency, amplitude

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

###############################################
# Test 1: Frequency Step.
###############################################
def TestFrequencyStep_Subplots():
    TimeN, XN, TrueFreqs = generateFreqStepSignal(Fs)

    # Compute results
    T_b1024, F_b1024 = freq_detection(XN, Fs, N=1024)
    T_b2048, F_b2048 = freq_detection(XN, Fs, N=2048)
    T_zp, F_zp = FreqDetectionZeroPad(XN, Fs, N=1024, padFactor=4)
    T_w, F_w = FreqDetectionWindowed(XN, Fs, N=1024)
    T_h, F_h = FreqDetectionHps(XN, Fs, N=1024, harmonics=3)

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Test 1: Frequency Step - Methods in Separate Subplots")

    # True Frequency
    axs[0, 0].plot(TimeN, TrueFreqs, 'b', label='True Frequency')
    axs[0, 0].set_title("True Frequency")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Frequency (Hz)")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Baseline (N=1024)
    axs[0, 1].plot(TimeN, TrueFreqs, 'b')
    axs[0, 1].plot(T_b1024, F_b1024, 'r', label='Baseline (N=1024)')
    axs[0, 1].set_title("Baseline (N=1024)")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Frequency (Hz)")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Baseline (N=2048)
    axs[0, 2].plot(TimeN, TrueFreqs, 'b')
    axs[0, 2].plot(T_b2048, F_b2048, 'g', label='Baseline (N=2048)')
    axs[0, 2].set_title("Baseline (N=2048)")
    axs[0, 2].set_xlabel("Time (s)")
    axs[0, 2].set_ylabel("Frequency (Hz)")
    axs[0, 2].grid(True)
    axs[0, 2].legend()

    # Zero-Pad
    axs[1, 0].plot(TimeN, TrueFreqs, 'b')
    axs[1, 0].plot(T_zp, F_zp, color='orange', label='Zero-Pad')
    axs[1, 0].set_title("Zero-Pad")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Windowed
    axs[1, 1].plot(TimeN, TrueFreqs, 'b')
    axs[1, 1].plot(T_w, F_w, color='purple', label='Windowed')
    axs[1, 1].set_title("Windowed")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Frequency (Hz)")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # HPS
    axs[1, 2].plot(TimeN, TrueFreqs, 'b')
    axs[1, 2].plot(T_h, F_h, color='brown', label='HPS')
    axs[1, 2].set_title("HPS")
    axs[1, 2].set_xlabel("Time (s)")
    axs[1, 2].set_ylabel("Frequency (Hz)")
    axs[1, 2].grid(True)
    axs[1, 2].legend()

    plt.tight_layout()
    plt.show()

###############################################
# Test 2: Pure Sine with Noise
###############################################
def TestPureSineNoise_Subplots():
    TimeN, CleanSignal, Frequency, Amplitude = generatePureSineNoise(Fs)
    SNRValues = np.logspace(-2, 2, 20)
    NoiseVars = (Amplitude**2 / (2 * SNRValues))

    AvgEst_b, Err_b, Std_b = [], [], []
    AvgEst_zp, Err_zp, Std_zp = [], [], []
    AvgEst_w, Err_w, Std_w = [], [], []
    AvgEst_h, Err_h, Std_h = [], [], []

    for NoiseVar in NoiseVars:
        Noise = np.random.normal(0, np.sqrt(NoiseVar), len(CleanSignal))
        XN = CleanSignal + Noise

        # Baseline
        _, Fb = freq_detection(XN, Fs, N=NDFT)
        Fb = np.array(Fb)
        eb = np.abs(Fb - Frequency)
        AvgEst_b.append(np.mean(Fb))
        Err_b.append(np.mean(eb))
        Std_b.append(np.std(eb))

        # Zero-Pad
        _, Fz = FreqDetectionZeroPad(XN, Fs, N=NDFT, padFactor=4)
        Fz = np.array(Fz)
        ez = np.abs(Fz - Frequency)
        AvgEst_zp.append(np.mean(Fz))
        Err_zp.append(np.mean(ez))
        Std_zp.append(np.std(ez))

        # Windowed
        _, Fw = FreqDetectionWindowed(XN, Fs, N=NDFT)
        Fw = np.array(Fw)
        ew = np.abs(Fw - Frequency)
        AvgEst_w.append(np.mean(Fw))
        Err_w.append(np.mean(ew))
        Std_w.append(np.std(ew))

        # HPS
        _, Fh = FreqDetectionHps(XN, Fs, N=NDFT, harmonics=3)
        Fh = np.array(Fh)
        eh = np.abs(Fh - Frequency)
        AvgEst_h.append(np.mean(Fh))
        Err_h.append(np.mean(eh))
        Std_h.append(np.std(eh))

    InvSNR = 1 / SNRValues

    fig, axs = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle("Test 2: Pure Sine with Noise - All Methods in Separate Subplots")

    # Baseline (Frequency Estimate)
    axs[0, 0].plot(InvSNR, AvgEst_b, 'r', label='Baseline')
    axs[0, 0].fill_between(InvSNR, np.array(AvgEst_b)-Std_b, np.array(AvgEst_b)+Std_b, color='r', alpha=0.2)
    axs[0, 0].axhline(Frequency, color='black', linestyle='--', label=f"True Freq = {Frequency}Hz")
    axs[0, 0].set_title("Baseline (Freq Est.)")
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel('1/SNR')
    axs[0, 0].set_ylabel('Frequency (Hz)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Zero-Pad (Frequency Estimate)
    axs[0, 1].plot(InvSNR, AvgEst_zp, 'g', label='Zero-Pad')
    axs[0, 1].fill_between(InvSNR, np.array(AvgEst_zp)-Std_zp, np.array(AvgEst_zp)+Std_zp, color='g', alpha=0.2)
    axs[0, 1].axhline(Frequency, color='black', linestyle='--')
    axs[0, 1].set_title("Zero-Pad (Freq Est.)")
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('1/SNR')
    axs[0, 1].set_ylabel('Frequency (Hz)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Windowed (Frequency Estimate)
    axs[0, 2].plot(InvSNR, AvgEst_w, color='purple', label='Windowed')
    axs[0, 2].fill_between(InvSNR, np.array(AvgEst_w)-Std_w, np.array(AvgEst_w)+Std_w, color='purple', alpha=0.2)
    axs[0, 2].axhline(Frequency, color='black', linestyle='--')
    axs[0, 2].set_title("Windowed (Freq Est.)")
    axs[0, 2].set_xscale('log')
    axs[0, 2].set_xlabel('1/SNR')
    axs[0, 2].set_ylabel('Frequency (Hz)')
    axs[0, 2].grid(True)
    axs[0, 2].legend()

    # HPS (Frequency Estimate)
    axs[0, 3].plot(InvSNR, AvgEst_h, color='orange', label='HPS')
    axs[0, 3].fill_between(InvSNR, np.array(AvgEst_h)-Std_h, np.array(AvgEst_h)+Std_h, color='orange', alpha=0.2)
    axs[0, 3].axhline(Frequency, color='black', linestyle='--')
    axs[0, 3].set_title("HPS (Freq Est.)")
    axs[0, 3].set_xscale('log')
    axs[0, 3].set_xlabel('1/SNR')
    axs[0, 3].set_ylabel('Frequency (Hz)')
    axs[0, 3].grid(True)
    axs[0, 3].legend()

    # Baseline (Error)
    axs[1, 0].plot(InvSNR, Err_b, 'r')
    axs[1, 0].set_title("Baseline (Error)")
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xlabel('1/SNR')
    axs[1, 0].set_ylabel('Error |f - f_hat| (Hz)')
    axs[1, 0].grid(True)

    # Zero-Pad (Error)
    axs[1, 1].plot(InvSNR, Err_zp, 'g')
    axs[1, 1].set_title("Zero-Pad (Error)")
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel('1/SNR')
    axs[1, 1].set_ylabel('Error |f - f_hat| (Hz)')
    axs[1, 1].grid(True)

    # Windowed (Error)
    axs[1, 2].plot(InvSNR, Err_w, color='purple')
    axs[1, 2].set_title("Windowed (Error)")
    axs[1, 2].set_xscale('log')
    axs[1, 2].set_yscale('log')
    axs[1, 2].set_xlabel('1/SNR')
    axs[1, 2].set_ylabel('Error |f - f_hat| (Hz)')
    axs[1, 2].grid(True)

    # HPS (Error)
    axs[1, 3].plot(InvSNR, Err_h, color='orange')
    axs[1, 3].set_title("HPS (Error)")
    axs[1, 3].set_xscale('log')
    axs[1, 3].set_yscale('log')
    axs[1, 3].set_xlabel('1/SNR')
    axs[1, 3].set_ylabel('Error |f - f_hat| (Hz)')
    axs[1, 3].grid(True)

    plt.tight_layout()
    plt.show()

###############################################
# Test 3: Real-world  Hello Jo S. Halle
###############################################
def TestRealWorld_Subplots(AudioFile='./SampleAudio/A4_oboe.wav'):
    TargetFs = 16000
    Duration = 2.2
    TrueFrequency = 440

    AudioData, SignalPower, N, TrueFrequency, TargetFs = loadAndPrepareAudio(AudioFile, TargetFs, Duration, TrueFrequency)
    NoisePowers = np.logspace(-2, 3, 20)

    AvgEst_b, Err_b, Var_b = [], [], []
    AvgEst_zp, Err_zp, Var_zp = [], [], []
    AvgEst_w, Err_w, Var_w = [], [], []
    AvgEst_h, Err_h, Var_h = [], [], []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise
        
        # Baseline
        _, Fb = freq_detection(NoisySignal, TargetFs, N=2048)
        Fb = np.clip(Fb, 0, TargetFs/2)
        eb = np.abs(Fb - TrueFrequency)
        AvgEst_b.append(np.mean(Fb))
        Err_b.append(np.mean(eb))
        Var_b.append(np.var(Fb))

        # Zero-Pad
        _, Fz = FreqDetectionZeroPad(NoisySignal, TargetFs, N=2048, padFactor=4)
        Fz = np.clip(Fz, 0, TargetFs/2)
        ez = np.abs(Fz - TrueFrequency)
        AvgEst_zp.append(np.mean(Fz))
        Err_zp.append(np.mean(ez))
        Var_zp.append(np.var(Fz))

        # Windowed
        _, Fw = FreqDetectionWindowed(NoisySignal, TargetFs, N=2048)
        Fw = np.clip(Fw, 0, TargetFs/2)
        ew = np.abs(Fw - TrueFrequency)
        AvgEst_w.append(np.mean(Fw))
        Err_w.append(np.mean(ew))
        Var_w.append(np.var(Fw))

        # HPS
        _, Fh = FreqDetectionHps(NoisySignal, TargetFs, N=2048, harmonics=3)
        Fh = np.clip(Fh, 0, TargetFs/2)
        eh = np.abs(Fh - TrueFrequency)
        AvgEst_h.append(np.mean(Fh))
        Err_h.append(np.mean(eh))
        Var_h.append(np.var(Fh))

    SNRValues = SignalPower / NoisePowers
    InvSNRValues = 1 / SNRValues

    Std_b = np.sqrt(Var_b)
    Std_zp = np.sqrt(Var_zp)
    Std_w = np.sqrt(Var_w)
    Std_h = np.sqrt(Var_h)

    fig, axs = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle(f"Test 3: Real-world ({AudioFile}) - All Methods in Separate Subplots")

    # Baseline (Frequency Estimate)
    axs[0, 0].plot(InvSNRValues, AvgEst_b, 'r', label='Baseline')
    axs[0, 0].fill_between(InvSNRValues, np.array(AvgEst_b)-Std_b, np.array(AvgEst_b)+Std_b, color='r', alpha=0.2)
    axs[0, 0].axhline(TrueFrequency, color='black', linestyle='--', label=f'True Freq={TrueFrequency}Hz')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title("Baseline (Freq Est.)")
    axs[0, 0].set_xlabel('1/SNR')
    axs[0, 0].set_ylabel('Frequency (Hz)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Zero-Pad (Frequency Estimate)
    axs[0, 1].plot(InvSNRValues, AvgEst_zp, 'g', label='Zero-Pad')
    axs[0, 1].fill_between(InvSNRValues, np.array(AvgEst_zp)-Std_zp, np.array(AvgEst_zp)+Std_zp, color='g', alpha=0.2)
    axs[0, 1].axhline(TrueFrequency, color='black', linestyle='--')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title("Zero-Pad (Freq Est.)")
    axs[0, 1].set_xlabel('1/SNR')
    axs[0, 1].set_ylabel('Frequency (Hz)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Windowed (Frequency Estimate)
    axs[0, 2].plot(InvSNRValues, AvgEst_w, color='purple', label='Windowed')
    axs[0, 2].fill_between(InvSNRValues, np.array(AvgEst_w)-Std_w, np.array(AvgEst_w)+Std_w, color='purple', alpha=0.2)
    axs[0, 2].axhline(TrueFrequency, color='black', linestyle='--')
    axs[0, 2].set_xscale('log')
    axs[0, 2].set_title("Windowed (Freq Est.)")
    axs[0, 2].set_xlabel('1/SNR')
    axs[0, 2].set_ylabel('Frequency (Hz)')
    axs[0, 2].grid(True)
    axs[0, 2].legend()

    # HPS (Frequency Estimate)
    axs[0, 3].plot(InvSNRValues, AvgEst_h, color='orange', label='HPS')
    axs[0, 3].fill_between(InvSNRValues, np.array(AvgEst_h)-Std_h, np.array(AvgEst_h)+Std_h, color='orange', alpha=0.2)
    axs[0, 3].axhline(TrueFrequency, color='black', linestyle='--')
    axs[0, 3].set_xscale('log')
    axs[0, 3].set_title("HPS (Freq Est.)")
    axs[0, 3].set_xlabel('1/SNR')
    axs[0, 3].set_ylabel('Frequency (Hz)')
    axs[0, 3].grid(True)
    axs[0, 3].legend()

    # Baseline (Error)
    axs[1, 0].plot(InvSNRValues, Err_b, 'r')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("Baseline (Error)")
    axs[1, 0].set_xlabel('1/SNR')
    axs[1, 0].set_ylabel('Error (Hz)')
    axs[1, 0].grid(True)

    # Zero-Pad (Error)
    axs[1, 1].plot(InvSNRValues, Err_zp, 'g')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title("Zero-Pad (Error)")
    axs[1, 1].set_xlabel('1/SNR')
    axs[1, 1].set_ylabel('Error (Hz)')
    axs[1, 1].grid(True)

    # Windowed (Error)
    axs[1, 2].plot(InvSNRValues, Err_w, color='purple')
    axs[1, 2].set_xscale('log')
    axs[1, 2].set_yscale('log')
    axs[1, 2].set_title("Windowed (Error)")
    axs[1, 2].set_xlabel('1/SNR')
    axs[1, 2].set_ylabel('Error (Hz)')
    axs[1, 2].grid(True)

    # HPS (Error)
    axs[1, 3].plot(InvSNRValues, Err_h, color='orange')
    axs[1, 3].set_xscale('log')
    axs[1, 3].set_yscale('log')
    axs[1, 3].set_title("HPS (Error)")
    axs[1, 3].set_xlabel('1/SNR')
    axs[1, 3].set_ylabel('Error (Hz)')
    axs[1, 3].grid(True)

    plt.tight_layout()
    plt.show()

###############################################
# Test 4: Vocal
###############################################
def TestVocal_Subplots(AudioFile='SampleAudio/Zauberflöte_vocal.wav', TargetFs=16000, Duration=2.2, TrueFrequency=440, NDFT=2048):
    AudioData, SignalPower, N, TrueFrequency, TargetFs = loadAndPrepareAudio(AudioFile, TargetFs, Duration, TrueFrequency)
    NoisePowers = np.logspace(-2, 3, 20)

    AvgEst_b, Err_b, Var_b = [], [], []
    AvgEst_zp, Err_zp, Var_zp = [], [], []
    AvgEst_w, Err_w, Var_w = [], [], []
    AvgEst_h, Err_h, Var_h = [], [], []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise

        # Baseline
        _, Fb = freq_detection(NoisySignal, TargetFs, N=NDFT)
        Fb = np.clip(Fb, 0, TargetFs/2)
        eb = np.abs(Fb - TrueFrequency)
        AvgEst_b.append(np.mean(Fb))
        Err_b.append(np.mean(eb))
        Var_b.append(np.var(Fb))

        # Zero-Pad
        _, Fz = FreqDetectionZeroPad(NoisySignal, TargetFs, N=NDFT, padFactor=4)
        Fz = np.clip(Fz, 0, TargetFs/2)
        ez = np.abs(Fz - TrueFrequency)
        AvgEst_zp.append(np.mean(Fz))
        Err_zp.append(np.mean(ez))
        Var_zp.append(np.var(Fz))

        # Windowed
        _, Fw = FreqDetectionWindowed(NoisySignal, TargetFs, N=NDFT)
        Fw = np.clip(Fw, 0, TargetFs/2)
        ew = np.abs(Fw - TrueFrequency)
        AvgEst_w.append(np.mean(Fw))
        Err_w.append(np.mean(ew))
        Var_w.append(np.var(Fw))

        # HPS
        _, Fh = FreqDetectionHps(NoisySignal, TargetFs, N=NDFT, harmonics=3)
        Fh = np.clip(Fh, 0, TargetFs/2)
        eh = np.abs(Fh - TrueFrequency)
        AvgEst_h.append(np.mean(Fh))
        Err_h.append(np.mean(eh))
        Var_h.append(np.var(Fh))

    SNRValues = SignalPower / NoisePowers
    InvSNRValues = 1 / SNRValues

    Std_b = np.sqrt(Var_b)
    Std_zp = np.sqrt(Var_zp)
    Std_w = np.sqrt(Var_w)
    Std_h = np.sqrt(Var_h)

    fig, axs = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle(f"Test 4: Vocal ({AudioFile}) - All Methods in Separate Subplots")

    # Baseline (Freq)
    axs[0, 0].plot(InvSNRValues, AvgEst_b, 'r')
    axs[0, 0].fill_between(InvSNRValues, np.array(AvgEst_b)-Std_b, np.array(AvgEst_b)+Std_b, color='r', alpha=0.2)
    axs[0, 0].axhline(TrueFrequency, color='black', linestyle='--', label=f'True={TrueFrequency}Hz')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_title("Baseline (Freq Est.)")
    axs[0, 0].set_xlabel('1/SNR')
    axs[0, 0].set_ylabel('Frequency (Hz)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # Zero-Pad (Freq)
    axs[0, 1].plot(InvSNRValues, AvgEst_zp, 'g')
    axs[0, 1].fill_between(InvSNRValues, np.array(AvgEst_zp)-Std_zp, np.array(AvgEst_zp)+Std_zp, color='g', alpha=0.2)
    axs[0, 1].axhline(TrueFrequency, color='black', linestyle='--')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_title("Zero-Pad (Freq Est.)")
    axs[0, 1].set_xlabel('1/SNR')
    axs[0, 1].set_ylabel('Frequency (Hz)')
    axs[0, 1].grid(True)

    # Windowed (Freq)
    axs[0, 2].plot(InvSNRValues, AvgEst_w, color='purple')
    axs[0, 2].fill_between(InvSNRValues, np.array(AvgEst_w)-Std_w, np.array(AvgEst_w)+Std_w, color='purple', alpha=0.2)
    axs[0, 2].axhline(TrueFrequency, color='black', linestyle='--')
    axs[0, 2].set_xscale('log')
    axs[0, 2].set_title("Windowed (Freq Est.)")
    axs[0, 2].set_xlabel('1/SNR')
    axs[0, 2].set_ylabel('Frequency (Hz)')
    axs[0, 2].grid(True)

    # HPS (Freq)
    axs[0, 3].plot(InvSNRValues, AvgEst_h, color='orange')
    axs[0, 3].fill_between(InvSNRValues, np.array(AvgEst_h)-Std_h, np.array(AvgEst_h)+Std_h, color='orange', alpha=0.2)
    axs[0, 3].axhline(TrueFrequency, color='black', linestyle='--')
    axs[0, 3].set_xscale('log')
    axs[0, 3].set_title("HPS (Freq Est.)")
    axs[0, 3].set_xlabel('1/SNR')
    axs[0, 3].set_ylabel('Frequency (Hz)')
    axs[0, 3].grid(True)

    # Baseline (Error)
    axs[1, 0].plot(InvSNRValues, Err_b, 'r')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("Baseline (Error)")
    axs[1, 0].set_xlabel('1/SNR')
    axs[1, 0].set_ylabel('Error (Hz)')
    axs[1, 0].grid(True)

    # Zero-Pad (Error)
    axs[1, 1].plot(InvSNRValues, Err_zp, 'g')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title("Zero-Pad (Error)")
    axs[1, 1].set_xlabel('1/SNR')
    axs[1, 1].set_ylabel('Error (Hz)')
    axs[1, 1].grid(True)

    # Windowed (Error)
    axs[1, 2].plot(InvSNRValues, Err_w, color='purple')
    axs[1, 2].set_xscale('log')
    axs[1, 2].set_yscale('log')
    axs[1, 2].set_title("Windowed (Error)")
    axs[1, 2].set_xlabel('1/SNR')
    axs[1, 2].set_ylabel('Error (Hz)')
    axs[1, 2].grid(True)

    # HPS (Error)
    axs[1, 3].plot(InvSNRValues, Err_h, color='orange')
    axs[1, 3].set_xscale('log')
    axs[1, 3].set_yscale('log')
    axs[1, 3].set_title("HPS (Error)")
    axs[1, 3].set_xlabel('1/SNR')
    axs[1, 3].set_ylabel('Error (Hz)')
    axs[1, 3].grid(True)

    plt.tight_layout()
    plt.show()

def TestCompositeAlgorithm():

    TimeN, XN, TrueFreqs = generateFreqStepSignal(Fs)

    # Compute results using the composite algorithm
    T_combined, F_combined = CombinedFreqDetection(XN, Fs, N=1024, padFactor=4, harmonics=3)

    # Compute results from individual methods (for comparison)
    T_zp, F_zp = FreqDetectionZeroPad(XN, Fs, N=1024, padFactor=4)
    T_w, F_w = FreqDetectionWindowed(XN, Fs, N=1024)
    T_h, F_h = FreqDetectionHps(XN, Fs, N=1024, harmonics=3)

    # Plot Results
    plt.figure(figsize=(12, 8))
    plt.title("Composite Algorithm: Combining Zero-Pad, Windowed, and HPS")

    # True Frequency
    plt.plot(TimeN, TrueFreqs, 'b', label="True Frequency")

    # Individual Methods
    plt.plot(T_zp, F_zp, 'orange', label="Zero-Pad")
    plt.plot(T_w, F_w, 'purple', label="Windowed")
    plt.plot(T_h, F_h, 'brown', label="HPS")

    # Combined Algorithm
    plt.plot(T_combined, F_combined, 'r', linewidth=2, label="Combined Algorithm")

    # Formatting
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def TestCombinedFreqDetectionStep_Subplots():
    TimeN, XN, TrueFreqs = generateFreqStepSignal(Fs)

    # Compute results using the composite algorithm
    T_combined, F_combined = CombinedFreqDetection(XN, Fs, N=1024, padFactor=4, harmonics=3)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Test 1: Frequency Step - Combined Algorithm")

    # True Frequency vs. Detected Frequency
    axs[0].plot(TimeN, TrueFreqs, 'b', label='True Frequency')
    axs[0].plot(T_combined, F_combined, 'r', label='Combined Algorithm')
    axs[0].set_title("Frequency Estimate")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].grid(True)
    axs[0].legend()

    # Error
    Error_combined = np.abs(F_combined - TrueFreqs[:len(F_combined)])
    axs[1].plot(T_combined, Error_combined, 'r')
    axs[1].set_title("Error")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Error (Hz)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def TestCombinedFreqDetectionSine_Subplots():
    TimeN, CleanSignal, Frequency, Amplitude = generatePureSineNoise(Fs)
    SNRValues = np.logspace(-2, 2, 20)
    NoiseVars = (Amplitude**2 / (2 * SNRValues))

    AvgEst_combined, Err_combined, Std_combined = [], [], []

    for NoiseVar in NoiseVars:
        Noise = np.random.normal(0, np.sqrt(NoiseVar), len(CleanSignal))
        XN = CleanSignal + Noise

        # Combined Algorithm
        _, F_combined = CombinedFreqDetection(XN, Fs, N=NDFT, padFactor=4, harmonics=3)
        F_combined = np.array(F_combined)
        e_combined = np.abs(F_combined - Frequency)
        AvgEst_combined.append(np.mean(F_combined))
        Err_combined.append(np.mean(e_combined))
        Std_combined.append(np.std(e_combined))

    InvSNR = 1 / SNRValues

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Test 2: Pure Sine with Noise - Combined Algorithm")

    # Frequency Estimate
    axs[0].plot(InvSNR, AvgEst_combined, 'r', label='Combined Algorithm')
    axs[0].fill_between(InvSNR, np.array(AvgEst_combined) - Std_combined, np.array(AvgEst_combined) + Std_combined, color='r', alpha=0.2)
    axs[0].axhline(Frequency, color='black', linestyle='--', label=f"True Frequency = {Frequency} Hz")
    axs[0].set_title("Frequency Estimate")
    axs[0].set_xscale('log')
    axs[0].set_xlabel("1/SNR")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].grid(True)
    axs[0].legend()

    # Error
    axs[1].plot(InvSNR, Err_combined, 'r')
    axs[1].set_title("Error")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("1/SNR")
    axs[1].set_ylabel("Error (Hz)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def TestCombinedFreqDetectionRealWorld(AudioFile='./SampleAudio/A4_oboe.wav'):
    TargetFs = 16000
    Duration = 2.2
    TrueFrequency = 440

    AudioData, SignalPower, N, TrueFrequency, TargetFs = loadAndPrepareAudio(AudioFile, TargetFs, Duration, TrueFrequency)
    NoisePowers = np.logspace(-2, 3, 20)

    AvgEst_combined, Err_combined, Var_combined = [], [], []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise

        # Combined Algorithm
        _, F_combined = CombinedFreqDetection(NoisySignal, TargetFs, N=2048, padFactor=4, harmonics=3)
        F_combined = np.clip(F_combined, 0, TargetFs / 2)
        e_combined = np.abs(F_combined - TrueFrequency)
        AvgEst_combined.append(np.mean(F_combined))
        Err_combined.append(np.mean(e_combined))
        Var_combined.append(np.var(F_combined))

    SNRValues = SignalPower / NoisePowers
    InvSNRValues = 1 / SNRValues
    Std_combined = np.sqrt(Var_combined)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Test 3: Real-world Audio ({AudioFile}) - Combined Algorithm")

    # Frequency Estimate
    axs[0].plot(InvSNRValues, AvgEst_combined, 'r', label='Combined Algorithm')
    axs[0].fill_between(InvSNRValues, np.array(AvgEst_combined) - Std_combined, np.array(AvgEst_combined) + Std_combined, color='r', alpha=0.2)
    axs[0].axhline(TrueFrequency, color='black', linestyle='--', label=f"True Frequency = {TrueFrequency} Hz")
    axs[0].set_title("Frequency Estimate")
    axs[0].set_xscale('log')
    axs[0].set_xlabel("1/SNR")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].grid(True)
    axs[0].legend()

    # Error
    axs[1].plot(InvSNRValues, Err_combined, 'r')
    axs[1].set_title("Error")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("1/SNR")
    axs[1].set_ylabel("Error (Hz)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def TestCombinedFreqDetectionRealWorldVocal(AudioFile='SampleAudio/Zauberflöte_vocal.wav'):
    TargetFs = 16000
    Duration = 2.2
    TrueFrequency = 440

    # Load and prepare audio
    AudioData, SignalPower, N, TrueFrequency, TargetFs = loadAndPrepareAudio(AudioFile, TargetFs, Duration, TrueFrequency)
    NoisePowers = np.logspace(-2, 3, 20)

    AvgEst_combined, Err_combined, Var_combined = [], [], []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise

        # Combined Algorithm
        _, F_combined = CombinedFreqDetection(NoisySignal, TargetFs, N=2048, padFactor=4, harmonics=3)
        F_combined = np.clip(F_combined, 0, TargetFs / 2)
        e_combined = np.abs(F_combined - TrueFrequency)
        AvgEst_combined.append(np.mean(F_combined))
        Err_combined.append(np.mean(e_combined))
        Var_combined.append(np.var(F_combined))

    SNRValues = SignalPower / NoisePowers
    InvSNRValues = 1 / SNRValues
    Std_combined = np.sqrt(Var_combined)

    # Create the plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Test 4: Real-world Audio (Vocal) - Combined Algorithm")

    # Frequency Estimate Plot
    axs[0].plot(InvSNRValues, AvgEst_combined, 'r', label='Combined Algorithm')
    axs[0].fill_between(InvSNRValues, np.array(AvgEst_combined) - Std_combined, np.array(AvgEst_combined) + Std_combined, color='r', alpha=0.2)
    axs[0].axhline(TrueFrequency, color='black', linestyle='--', label=f"True Frequency = {TrueFrequency} Hz")
    axs[0].set_title("Frequency Estimate")
    axs[0].set_xscale('log')
    axs[0].set_xlabel("1/SNR")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].grid(True)
    axs[0].legend()

    # Error Plot
    axs[1].plot(InvSNRValues, Err_combined, 'r')
    axs[1].set_title("Error")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("1/SNR")
    axs[1].set_ylabel("Error (Hz)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()



###############################################
# Run All Tests
###############################################
if __name__ == "__main__":
    # Test 1: Frequency Step
    TestFrequencyStep_Subplots()
    
    # Test 2: Pure Sine with Noise
    TestPureSineNoise_Subplots()
    
    # Test 3: Real-world
    TestRealWorld_Subplots('./SampleAudio/A4_oboe.wav')
    
    # Test 4: Vocal
    TestVocal_Subplots('SampleAudio/Zauberflöte_vocal.wav')

    # Test 5: Composite Algorithm
    TestCompositeAlgorithm()

    # Test 6: Combined Algorithm - Frequency Step
    TestCombinedFreqDetectionStep_Subplots()

    # Test 7: Combined Algorithm - Pure Sine with Noise
    TestCombinedFreqDetectionSine_Subplots()

    # Test 8: Combined Algorithm - Real-world
    TestCombinedFreqDetectionRealWorld('./SampleAudio/A4_oboe.wav')

    # Test 9: Combined Algorithm - Vocal
    TestCombinedFreqDetectionRealWorldVocal('SampleAudio/Zauberflöte_vocal.wav')

    print("All tests completed successfully!")