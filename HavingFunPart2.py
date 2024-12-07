import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from baseline_algorithm import freq_detection

# Import improved frequency detection methods
from FreqDetection import freqDetectionZeroPad, FreqDetectionWindowed, FreqDetectionHps

###############################################
# Parameters for All Tests
###############################################
Fs = 10_000   # Sampling Frequency for Synthetic Tests
NDFT = 1024   # Default DFT Size for Baseline Tests

###############################################
# Helper Function: Generate Frequency Step Signal
###############################################
def generate_freq_step_signal(Fs, TTotal=2.0, F1=440, F2=880, Amplitude=1, NoiseVar=0.5):
    NTotal = int(TTotal * Fs)
    TimeN = np.arange(NTotal) / Fs
    TimeChange = TTotal / 2
    NChange = int(TimeChange * Fs)

    Signal = np.zeros(NTotal)
    Signal[:NChange] = Amplitude * np.sin(2 * np.pi * F1 * TimeN[:NChange])
    Signal[NChange:] = Amplitude * np.sin(2 * np.pi * F2 * TimeN[NChange:])
    
    Noise = np.random.normal(scale=np.sqrt(NoiseVar), size=NTotal)
    XN = Signal + Noise
    TrueFreqs = np.where(TimeN < TimeChange, F1, F2)
    return TimeN, XN, TrueFreqs

###############################################
# Test 1: Frequency Step Response (Baseline)
###############################################
def TestFrequencyStep():
    TimeN, XN, TrueFreqs = generate_freq_step_signal(Fs)
    NValues = [1024, 2048]

    plt.figure(figsize=(10, 5))
    plt.plot(TimeN, TrueFreqs, label='True Frequency', color='blue')
    
    for NValue in NValues:
        Timestamps, Freqs = freq_detection(XN, Fs, N=NValue)
        plt.plot(Timestamps, Freqs, label=f'Baseline (N={NValue})')
    
    plt.title("Test 1: Frequency Step Response (Baseline)")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################
# Test 1 Variations with Improved Methods
###############################################
def TestFrequencyStep_ZeroPad():
    TimeN, XN, TrueFreqs = generate_freq_step_signal(Fs)
    plt.figure(figsize=(10, 5))
    plt.plot(TimeN, TrueFreqs, label='True Frequency', color='blue')
    
    # Zero-padding approach
    Timestamps, Freqs = freqDetectionZeroPad(XN, Fs, N=1024, padFactor=4)
    plt.plot(Timestamps, Freqs, label='Zero-Pad')
    
    plt.title("Test 1: Frequency Step Response (Zero-Pad)")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

def TestFrequencyStep_Windowed():
    TimeN, XN, TrueFreqs = generate_freq_step_signal(Fs)
    plt.figure(figsize=(10, 5))
    plt.plot(TimeN, TrueFreqs, label='True Frequency', color='blue')
    
    # Windowed approach
    Timestamps, Freqs = FreqDetectionWindowed(XN, Fs, N=1024)
    plt.plot(Timestamps, Freqs, label='Windowed')
    
    plt.title("Test 1: Frequency Step Response (Windowed)")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

def TestFrequencyStep_Hps():
    TimeN, XN, TrueFreqs = generate_freq_step_signal(Fs)
    plt.figure(figsize=(10, 5))
    plt.plot(TimeN, TrueFreqs, label='True Frequency', color='blue')
    
    # HPS approach
    Timestamps, Freqs = FreqDetectionHps(XN, Fs, N=1024, harmonics=3)
    plt.plot(Timestamps, Freqs, label='HPS')
    
    plt.title("Test 1: Frequency Step Response (HPS)")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################
# Test 2: Pure Sine with Noise (Baseline)
###############################################
def TestPureSineNoise():
    Duration = 4.0
    N = int(Duration * Fs)
    Frequency = 885
    Amplitude = 1.0
    TimeN = np.arange(N) / Fs
    CleanSignal = Amplitude * np.sin(2 * np.pi * Frequency * TimeN)

    SNRValues = np.logspace(-2, 2, 20)
    NoiseVars = (Amplitude**2 / (2 * SNRValues))

    AvgEstimates = []
    AvgErrors = []
    ErrorBars = []

    for NoiseVar in NoiseVars:
        Noise = np.random.normal(0, np.sqrt(NoiseVar), N)
        XN = CleanSignal + Noise
        Timestamps, Freqs = freq_detection(XN, Fs, N=NDFT)
        Freqs = np.array(Freqs)

        AvgFreqEstimate = np.mean(Freqs)
        AvgEstimates.append(AvgFreqEstimate)

        Errors = np.abs(Freqs - Frequency)
        AvgError = np.mean(Errors)
        AvgErrors.append(AvgError)

        ErrorBars.append(np.std(Errors))

    InvSNR = 1 / SNRValues
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(InvSNR, AvgEstimates, label='Baseline Avg. Freq. Estimate', color='red')
    plt.fill_between(InvSNR, 
                     np.array(AvgEstimates) - np.array(ErrorBars), 
                     np.array(AvgEstimates) + np.array(ErrorBars), 
                     color='red', alpha=0.2)
    plt.axhline(Frequency, color='black', linestyle='--', label=f'True Frequency = {Frequency} Hz')
    plt.xscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Frequency Estimate (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Baseline Frequency Estimates")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(InvSNR, AvgErrors, label='Baseline Avg. Error', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Error |f - f_hat| (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Baseline Estimation Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

###############################################
# Test 2 Variations with Improved Methods
###############################################
def TestPureSineNoise_ZeroPad():
    Duration = 4.0
    N = int(Duration * Fs)
    Frequency = 885
    Amplitude = 1.0
    TimeN = np.arange(N) / Fs
    CleanSignal = Amplitude * np.sin(2 * np.pi * Frequency * TimeN)

    SNRValues = np.logspace(-2, 2, 20)
    NoiseVars = (Amplitude**2 / (2 * SNRValues))

    AvgEstimates = []
    AvgErrors = []
    ErrorBars = []

    for NoiseVar in NoiseVars:
        Noise = np.random.normal(0, np.sqrt(NoiseVar), N)
        XN = CleanSignal + Noise
        Timestamps, Freqs = freqDetectionZeroPad(XN, Fs, N=NDFT, padFactor=4)
        Freqs = np.array(Freqs)

        AvgFreqEstimate = np.mean(Freqs)
        AvgEstimates.append(AvgFreqEstimate)

        Errors = np.abs(Freqs - Frequency)
        AvgError = np.mean(Errors)
        AvgErrors.append(AvgError)
        ErrorBars.append(np.std(Errors))

    InvSNR = 1 / SNRValues
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(InvSNR, AvgEstimates, label='Zero-Pad Avg. Freq. Estimate', color='green')
    plt.fill_between(InvSNR, 
                     np.array(AvgEstimates) - np.array(ErrorBars), 
                     np.array(AvgEstimates) + np.array(ErrorBars), 
                     color='green', alpha=0.2)
    plt.axhline(Frequency, color='black', linestyle='--', label=f'True Frequency = {Frequency} Hz')
    plt.xscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Frequency Estimate (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Zero-Pad Frequency Estimates")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(InvSNR, AvgErrors, label='Zero-Pad Avg. Error', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Error |f - f_hat| (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Zero-Pad Estimation Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def TestPureSineNoise_Windowed():
    # Similar to TestPureSineNoise_ZeroPad, but calling FreqDetectionWindowed
    # and updating labels accordingly.
    Duration = 4.0
    N = int(Duration * Fs)
    Frequency = 885
    Amplitude = 1.0
    TimeN = np.arange(N) / Fs
    CleanSignal = Amplitude * np.sin(2 * np.pi * Frequency * TimeN)

    SNRValues = np.logspace(-2, 2, 20)
    NoiseVars = (Amplitude**2 / (2 * SNRValues))

    AvgEstimates = []
    AvgErrors = []
    ErrorBars = []

    for NoiseVar in NoiseVars:
        Noise = np.random.normal(0, np.sqrt(NoiseVar), N)
        XN = CleanSignal + Noise
        Timestamps, Freqs = FreqDetectionWindowed(XN, Fs, N=NDFT)
        Freqs = np.array(Freqs)

        AvgFreqEstimate = np.mean(Freqs)
        AvgEstimates.append(AvgFreqEstimate)

        Errors = np.abs(Freqs - Frequency)
        AvgError = np.mean(Errors)
        AvgErrors.append(AvgError)
        ErrorBars.append(np.std(Errors))

    InvSNR = 1 / SNRValues
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(InvSNR, AvgEstimates, label='Windowed Avg. Freq. Estimate', color='purple')
    plt.fill_between(InvSNR, 
                     np.array(AvgEstimates) - np.array(ErrorBars), 
                     np.array(AvgEstimates) + np.array(ErrorBars), 
                     color='purple', alpha=0.2)
    plt.axhline(Frequency, color='black', linestyle='--', label=f'True Frequency = {Frequency} Hz')
    plt.xscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Frequency Estimate (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Windowed Frequency Estimates")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(InvSNR, AvgErrors, label='Windowed Avg. Error', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Error |f - f_hat| (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Windowed Estimation Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def TestPureSineNoise_Hps():
    # Similar to TestPureSineNoise_ZeroPad, but calling FreqDetectionHps
    Duration = 4.0
    N = int(Duration * Fs)
    Frequency = 885
    Amplitude = 1.0
    TimeN = np.arange(N) / Fs
    CleanSignal = Amplitude * np.sin(2 * np.pi * Frequency * TimeN)

    SNRValues = np.logspace(-2, 2, 20)
    NoiseVars = (Amplitude**2 / (2 * SNRValues))

    AvgEstimates = []
    AvgErrors = []
    ErrorBars = []

    for NoiseVar in NoiseVars:
        Noise = np.random.normal(0, np.sqrt(NoiseVar), N)
        XN = CleanSignal + Noise
        Timestamps, Freqs = FreqDetectionHps(XN, Fs, N=NDFT, harmonics=3)
        Freqs = np.array(Freqs)

        AvgFreqEstimate = np.mean(Freqs)
        AvgEstimates.append(AvgFreqEstimate)

        Errors = np.abs(Freqs - Frequency)
        AvgError = np.mean(Errors)
        AvgErrors.append(AvgError)
        ErrorBars.append(np.std(Errors))

    InvSNR = 1 / SNRValues
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(InvSNR, AvgEstimates, label='HPS Avg. Freq. Estimate', color='orange')
    plt.fill_between(InvSNR, 
                     np.array(AvgEstimates) - np.array(ErrorBars), 
                     np.array(AvgEstimates) + np.array(ErrorBars), 
                     color='orange', alpha=0.2)
    plt.axhline(Frequency, color='black', linestyle='--', label=f'True Frequency = {Frequency} Hz')
    plt.xscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Frequency Estimate (Hz)')
    plt.title("Test 2: Pure Sine with Noise - HPS Frequency Estimates")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(InvSNR, AvgErrors, label='HPS Avg. Error', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Error |f - f_hat| (Hz)')
    plt.title("Test 2: Pure Sine with Noise - HPS Estimation Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

###############################################
# Test 3 Variations: Real-world Audio with Noise
###############################################

def TestRealWorldNoise(AudioFile='./SampleAudio/A4_oboe.wav'):
    """
    Load a real-world audio file (such as a musical instrument note),
    resample to a chosen Fs, and add varying noise levels to assess how
    the algorithm performs on complex waveforms under noise.
    """
    # Desired Parameters
    TargetFs = 16_000   # Choose a Suitable Sampling Rate for Analysis
    Duration = 2.2
    N = int(Duration * TargetFs)
    TrueFrequency = 440 # Expected Fundamental Frequency for A4
    
    # Load and Preprocess Audio
    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize
    
    # Resample if Needed
    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)
    
    # Ensure We Have N Samples (Pad or Truncate)
    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]
        
    # Calculate Signal Power
    SignalPower = np.mean(AudioData**2)
    
    # Define Noise Powers and Arrays to Store Results
    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []
    
    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise
        
        # Frequency Detection
        _, Freqs = freq_detection(NoisySignal, TargetFs, N=2048)
        Freqs = np.clip(Freqs, 0, TargetFs/2)
        
        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)
        
        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)
        
        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)
    
    InvSNRValues = 1 / np.array(SNRValues)
    
    # Plot Results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(InvSNRValues, AvgEstimates, color='r')
    # Show ±1 Std Around the Mean Estimate
    StdDevs = np.sqrt(Variances)
    ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='r', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_title(f"Test 3: Real-world Signal - Average Frequency Estimate with Noise\n({AudioFile})")
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)
    
    ax2.plot(InvSNRValues, AvgErrors, color='b')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("Test 3: Real-world Signal - Estimation Error vs. Noise")
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error |f - f_hat| (Hz)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def TestRealWorldNoise_ZeroPad(AudioFile='./SampleAudio/A4_oboe.wav'):
    TargetFs = 16_000
    Duration = 2.2
    N = int(Duration * TargetFs)
    TrueFrequency = 440

    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize

    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)

    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]

    SignalPower = np.mean(AudioData**2)

    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise
        
        # Using zero-padding method here:
        _, Freqs = freqDetectionZeroPad(NoisySignal, TargetFs, N=2048, padFactor=4)
        Freqs = np.clip(Freqs, 0, TargetFs/2)

        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)

        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)

        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)

    InvSNRValues = 1 / np.array(SNRValues)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(InvSNRValues, AvgEstimates, color='g')
    StdDevs = np.sqrt(Variances)
    ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='g', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_title(f"Test 3: Real-world (Zero-Pad) - Avg Frequency w/ Noise\n({AudioFile})")
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)

    ax2.plot(InvSNRValues, AvgErrors, color='b')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("Test 3: Real-world (Zero-Pad) - Estimation Error vs. Noise")
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error |f - f_hat| (Hz)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def TestRealWorldNoise_Windowed(AudioFile='./SampleAudio/A4_oboe.wav'):
    TargetFs = 16_000
    Duration = 2.2
    N = int(Duration * TargetFs)
    TrueFrequency = 440

    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize

    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)

    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]

    SignalPower = np.mean(AudioData**2)

    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise
        
        # Using windowed method
        _, Freqs = FreqDetectionWindowed(NoisySignal, TargetFs, N=2048)
        Freqs = np.clip(Freqs, 0, TargetFs/2)

        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)

        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)

        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)

    InvSNRValues = 1 / np.array(SNRValues)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(InvSNRValues, AvgEstimates, color='purple')
    StdDevs = np.sqrt(Variances)
    ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='purple', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_title(f"Test 3: Real-world (Windowed) - Avg Frequency w/ Noise\n({AudioFile})")
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)

    ax2.plot(InvSNRValues, AvgErrors, color='b')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("Test 3: Real-world (Windowed) - Estimation Error vs. Noise")
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error |f - f_hat| (Hz)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def TestRealWorldNoise_Hps(AudioFile='./SampleAudio/A4_oboe.wav'):
    TargetFs = 16_000
    Duration = 2.2
    N = int(Duration * TargetFs)
    TrueFrequency = 440

    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize

    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)

    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]

    SignalPower = np.mean(AudioData**2)

    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise
        
        # Using HPS method
        _, Freqs = FreqDetectionHps(NoisySignal, TargetFs, N=2048, harmonics=3)
        Freqs = np.clip(Freqs, 0, TargetFs/2)

        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)

        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)

        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)

    InvSNRValues = 1 / np.array(SNRValues)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(InvSNRValues, AvgEstimates, color='orange')
    StdDevs = np.sqrt(Variances)
    ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='orange', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_title(f"Test 3: Real-world (HPS) - Avg Frequency w/ Noise\n({AudioFile})")
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)

    ax2.plot(InvSNRValues, AvgErrors, color='b')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("Test 3: Real-world (HPS) - Estimation Error vs. Noise")
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error |f - f_hat| (Hz)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


###############################################
# Test 4 Variations: Vocal Audio with Noise
###############################################

def TestVocalNoise(AudioFile='SampleAudio/Zauberflöte_vocal.wav',
                   TargetFs=16000, Duration=2.2, TrueFrequency=440, NDFT=2048):
    """
    Analyze a vocal audio file, adding varying levels of noise, and 
    evaluate frequency detection accuracy.
    """
    # Load and Preprocess Audio
    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize
    
    N = int(Duration * TargetFs)
    
    # Resample if Needed
    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)
    
    # Ensure We Have N Samples (Pad or Truncate)
    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]
        
    # Calculate Signal Power
    SignalPower = np.mean(AudioData**2)
    
    # Define Noise Powers and Arrays to Store Results
    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []
    
    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise
        
        # Frequency Detection
        _, Freqs = freq_detection(NoisySignal, TargetFs, N=NDFT)
        
        # Clip Frequencies to a Reasonable Range
        Freqs = np.clip(Freqs, 0, TargetFs/2)
        
        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)
        
        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)
        
        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)
    
    InvSNRValues = 1 / np.array(SNRValues)
    
    # Plot Results
    fig, (Ax1, Ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    Ax1.plot(InvSNRValues, AvgEstimates, color='r')
    StdDevs = np.sqrt(Variances)
    Ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='r', alpha=0.2)
    Ax1.set_xscale('log')
    Ax1.set_title(f"Vocal Test: Average Frequency Estimate with Noise\n({AudioFile})")
    Ax1.set_ylabel('Frequency Estimate (Hz)')
    Ax1.grid(True)
    
    Ax2.plot(InvSNRValues, AvgErrors, color='b')
    Ax2.set_xscale('log')
    Ax2.set_yscale('log')
    Ax2.set_title("Vocal Test: Estimation Error vs. Noise")
    Ax2.set_xlabel('1/SNR')
    Ax2.set_ylabel('Error |f - f_hat| (Hz)')
    Ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def TestVocalNoise_ZeroPad(AudioFile='SampleAudio/Zauberflöte_vocal.wav',
                           TargetFs=16000, Duration=2.2, TrueFrequency=440, NDFT=2048):
    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize

    N = int(Duration * TargetFs)

    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)

    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]

    SignalPower = np.mean(AudioData**2)

    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise

        # Zero-pad method
        _, Freqs = freqDetectionZeroPad(NoisySignal, TargetFs, N=NDFT, padFactor=4)
        Freqs = np.clip(Freqs, 0, TargetFs/2)

        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)

        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)

        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)

    InvSNRValues = 1 / np.array(SNRValues)
    fig, (Ax1, Ax2) = plt.subplots(2, 1, figsize=(10, 10))

    Ax1.plot(InvSNRValues, AvgEstimates, color='g')
    StdDevs = np.sqrt(Variances)
    Ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='g', alpha=0.2)
    Ax1.set_xscale('log')
    Ax1.set_title(f"Vocal Test (Zero-Pad): Average Frequency Estimate with Noise\n({AudioFile})")
    Ax1.set_ylabel('Frequency Estimate (Hz)')
    Ax1.grid(True)

    Ax2.plot(InvSNRValues, AvgErrors, color='b')
    Ax2.set_xscale('log')
    Ax2.set_yscale('log')
    Ax2.set_title("Vocal Test (Zero-Pad): Estimation Error vs. Noise")
    Ax2.set_xlabel('1/SNR')
    Ax2.set_ylabel('Error |f - f_hat| (Hz)')
    Ax2.grid(True)

    plt.tight_layout()
    plt.show()


def TestVocalNoise_Windowed(AudioFile='SampleAudio/Zauberflöte_vocal.wav',
                            TargetFs=16000, Duration=2.2, TrueFrequency=440, NDFT=2048):
    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize

    N = int(Duration * TargetFs)

    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)

    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]

    SignalPower = np.mean(AudioData**2)

    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise

        # Windowed method
        _, Freqs = FreqDetectionWindowed(NoisySignal, TargetFs, N=NDFT)
        Freqs = np.clip(Freqs, 0, TargetFs/2)

        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)

        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)

        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)

    InvSNRValues = 1 / np.array(SNRValues)
    fig, (Ax1, Ax2) = plt.subplots(2, 1, figsize=(10, 10))

    Ax1.plot(InvSNRValues, AvgEstimates, color='purple')
    StdDevs = np.sqrt(Variances)
    Ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='purple', alpha=0.2)
    Ax1.set_xscale('log')
    Ax1.set_title(f"Vocal Test (Windowed): Average Frequency Estimate with Noise\n({AudioFile})")
    Ax1.set_ylabel('Frequency Estimate (Hz)')
    Ax1.grid(True)

    Ax2.plot(InvSNRValues, AvgErrors, color='b')
    Ax2.set_xscale('log')
    Ax2.set_yscale('log')
    Ax2.set_title("Vocal Test (Windowed): Estimation Error vs. Noise")
    Ax2.set_xlabel('1/SNR')
    Ax2.set_ylabel('Error |f - f_hat| (Hz)')
    Ax2.grid(True)

    plt.tight_layout()
    plt.show()


def TestVocalNoise_Hps(AudioFile='SampleAudio/Zauberflöte_vocal.wav',
                       TargetFs=16000, Duration=2.2, TrueFrequency=440, NDFT=2048):
    OrigFs, AudioData = wavfile.read(AudioFile)
    AudioData = AudioData.astype(np.float32) / np.max(np.abs(AudioData))  # Normalize

    N = int(Duration * TargetFs)

    if OrigFs != TargetFs:
        AudioData = resample_poly(AudioData, TargetFs, OrigFs)

    if len(AudioData) < N:
        AudioData = np.pad(AudioData, (0, N - len(AudioData)), 'constant')
    else:
        AudioData = AudioData[:N]

    SignalPower = np.mean(AudioData**2)

    NoisePowers = np.logspace(-2, 2, 20)
    AvgEstimates = []
    AvgErrors = []
    Variances = []
    SNRValues = []

    for NoisePower in NoisePowers:
        Noise = np.random.normal(scale=np.sqrt(NoisePower), size=len(AudioData))
        NoisySignal = AudioData + Noise

        # HPS method
        _, Freqs = FreqDetectionHps(NoisySignal, TargetFs, N=NDFT, harmonics=3)
        Freqs = np.clip(Freqs, 0, TargetFs/2)

        AvgFreqEstimate = np.mean(Freqs)
        Var = np.var(Freqs)
        Error = np.abs(AvgFreqEstimate - TrueFrequency)

        SNR = SignalPower / NoisePower
        SNRValues.append(SNR)

        AvgEstimates.append(AvgFreqEstimate)
        AvgErrors.append(Error)
        Variances.append(Var)

    InvSNRValues = 1 / np.array(SNRValues)
    fig, (Ax1, Ax2) = plt.subplots(2, 1, figsize=(10, 10))

    Ax1.plot(InvSNRValues, AvgEstimates, color='orange')
    StdDevs = np.sqrt(Variances)
    Ax1.fill_between(InvSNRValues, 
                     np.array(AvgEstimates) - StdDevs, 
                     np.array(AvgEstimates) + StdDevs, 
                     color='orange', alpha=0.2)
    Ax1.set_xscale('log')
    Ax1.set_title(f"Vocal Test (HPS): Average Frequency Estimate with Noise\n({AudioFile})")
    Ax1.set_ylabel('Frequency Estimate (Hz)')
    Ax1.grid(True)

    Ax2.plot(InvSNRValues, AvgErrors, color='b')
    Ax2.set_xscale('log')
    Ax2.set_yscale('log')
    Ax2.set_title("Vocal Test (HPS): Estimation Error vs. Noise")
    Ax2.set_xlabel('1/SNR')
    Ax2.set_ylabel('Error |f - f_hat| (Hz)')
    Ax2.grid(True)

    plt.tight_layout()
    plt.show()

###############################################
# Run All Tests
###############################################
if __name__ == "__main__":
    # Baseline Tests
    TestFrequencyStep()
    TestPureSineNoise()
    TestRealWorldNoise('./SampleAudio/A4_oboe.wav')
    TestVocalNoise()

    # Improved Method Tests: 
    # Frequency Step
    TestFrequencyStep_ZeroPad()
    TestFrequencyStep_Windowed()
    TestFrequencyStep_Hps()
    
    # Pure Sine Noise
    TestPureSineNoise_ZeroPad()
    TestPureSineNoise_Windowed()
    TestPureSineNoise_Hps()
    
    # Improved Tests for Real-world audio
    TestRealWorldNoise_ZeroPad('./SampleAudio/A4_oboe.wav')
    TestRealWorldNoise_Windowed('./SampleAudio/A4_oboe.wav')
    TestRealWorldNoise_Hps('./SampleAudio/A4_oboe.wav')

    # Improved Tests for Vocal audio
    TestVocalNoise_ZeroPad('SampleAudio/Zauberflöte_vocal.wav')
    TestVocalNoise_Windowed('SampleAudio/Zauberflöte_vocal.wav')
    TestVocalNoise_Hps('SampleAudio/Zauberflöte_vocal.wav')
