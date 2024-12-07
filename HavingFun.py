import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from baseline_algorithm import freq_detection
from FreqDetection import freqDetectionZeroPad, FreqDetectionWindowed, FreqDetectionHps

###############################################
# Parameters for All Tests
###############################################
Fs = 10_000   # Sampling Frequency for Synthetic Tests
NDFT = 1024  # Default DFT Size for Baseline Tests


###############################################
# Test 1: Frequency Step Response
###############################################
def TestFrequencyStep():
    """
    Generate a signal that changes frequency abruptly at t = TTotal/2,
    add noise, and feed it into the baseline frequency detection 
    algorithm. The goal is to assess how quickly the frequency detection 
    adapts to a sudden change in frequency.
    """
    # Test Parameters
    TTotal = 2.0  # Total Duration in Seconds
    NTotal = int(TTotal * Fs)
    F1 = 440       # Initial Frequency
    F2 = 880       # Frequency After Step
    Amplitude = 1          # Signal Amplitude
    NoiseVar = 0.5
    
    # Time Vector
    TimeN = np.arange(NTotal) / Fs
    TimeChange = TTotal / 2
    NChange = int(TimeChange * Fs)
    
    # Construct the Piecewise Signal
    Signal = np.zeros(NTotal)
    Signal[:NChange] = Amplitude * np.sin(2 * np.pi * F1 * TimeN[:NChange])
    Signal[NChange:] = Amplitude * np.sin(2 * np.pi * F2 * TimeN[NChange:])
    
    # Add Noise
    Noise = np.random.normal(scale=np.sqrt(NoiseVar), size=NTotal)
    XN = Signal + Noise
    
    # True Frequency Over Time
    TrueFreqs = np.where(TimeN < TimeChange, F1, F2)
    
    # Test Different DFT Window Sizes
    NValues = [1024, 2048]
    
    plt.figure(figsize=(10, 5))
    plt.plot(TimeN, TrueFreqs, label='True Frequency $f(t)$', color='blue')
    
    for NValue in NValues:
        Timestamps, Freqs = freq_detection(XN, Fs, N=NValue)
        plt.plot(Timestamps, Freqs, label=f'Estimated Frequency (N={NValue})')
    
    plt.title("Test 1: Frequency Step Response")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()


###############################################
# Test 2: Pure Sine with Noise
###############################################
def TestPureSineNoise():
    """
    Generate a pure sine wave signal and add varying levels of white 
    noise. Evaluate how the average estimated frequency and estimation 
    error change as a function of SNR.
    """
    # Test Parameters
    Duration = 4.0        # Duration of the Test Signal
    N = int(Duration * Fs)
    Frequency = 885       # True Signal Frequency
    Amplitude = 1.0       # Amplitude of the Sine Wave
    
    # Generate the Clean Sine Wave
    TimeN = np.arange(N) / Fs
    CleanSignal = Amplitude * np.sin(2 * np.pi * Frequency * TimeN)
    
    # Define a Range of SNR Values
    SNRValues = np.logspace(-2, 2, 20)
    NoiseVars = (Amplitude**2 / (2 * SNRValues))
    
    AvgEstimates = []
    AvgErrors = []
    ErrorBars = []
    
    for NoiseVar in NoiseVars:
        # Add Noise
        Noise = np.random.normal(0, np.sqrt(NoiseVar), N)
        XN = CleanSignal + Noise
        
        # Run Frequency Detection
        Timestamps, Freqs = freq_detection(XN, Fs, N=NDFT)
        Freqs = np.array(Freqs)
        
        AvgFreqEstimate = np.mean(Freqs)
        AvgEstimates.append(AvgFreqEstimate)
        
        Errors = np.abs(Freqs - Frequency)
        AvgError = np.mean(Errors)
        AvgErrors.append(AvgError)
        
        ErrorBars.append(np.std(Errors))
    
    # Plot Results
    plt.figure(figsize=(10, 8))
    
    # Plot Average Frequency vs. 1/SNR
    InvSNR = 1 / SNRValues
    
    plt.subplot(2, 1, 1)
    plt.plot(InvSNR, AvgEstimates, label='Average Frequency Estimate', color='red')
    plt.fill_between(InvSNR, 
                     np.array(AvgEstimates) - np.array(ErrorBars), 
                     np.array(AvgEstimates) + np.array(ErrorBars), 
                     color='red', alpha=0.2)
    plt.axhline(Frequency, color='black', linestyle='--', label=f'True Frequency = {Frequency} Hz')
    plt.xscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Frequency Estimate (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Frequency Estimates")
    plt.legend()
    plt.grid(True)
    
    # Plot Average Error vs. 1/SNR
    plt.subplot(2, 1, 2)
    plt.plot(InvSNR, AvgErrors, label='Average Error', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Error |f - f_hat| (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Estimation Error")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


###############################################
# Test 3: Real-world Signal with Noise
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

###############################################
# Test 4: Vocal Audio with Noise
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

###############################################
# Run All Tests
###############################################
if __name__ == "__main__":
    # Test 1: Step in Frequency
    TestFrequencyStep()
    
    # Test 2: Pure Sine Wave with Noise
    TestPureSineNoise()
    
    # Test 3: Real-world Audio with Noise
    TestRealWorldNoise('./SampleAudio/A4_oboe.wav')

    # Test 4: Vocal Audio with Noise
    TestVocalNoise()

    
