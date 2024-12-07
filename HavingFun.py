import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly
from baseline_algorithm import freq_detection

###############################################
# Parameters for all tests
###############################################
fs = 10_000   # Sampling frequency for synthetic tests
N_DFT = 1024  # Default DFT size for baseline tests


###############################################
# Test 1: Frequency Step Response
###############################################
def test_frequency_step():
    """
    Generate a signal that changes frequency abruptly at t = T_total/2,
    add noise, and feed it into the baseline frequency detection 
    algorithm. The goal is to assess how quickly the frequency detection 
    adapts to a sudden change in frequency.
    """
    # Test parameters
    T_total = 2.0  # Total duration in seconds
    N_total = int(T_total * fs)
    f1 = 440       # Initial frequency
    f2 = 880       # Frequency after step
    A = 1          # Signal amplitude
    noise_var = 0.5
    
    # Time vector
    t_n = np.arange(N_total) / fs
    t_change = T_total / 2
    n_change = int(t_change * fs)
    
    # Construct the piecewise signal
    signal = np.zeros(N_total)
    signal[:n_change] = A * np.sin(2 * np.pi * f1 * t_n[:n_change])
    signal[n_change:] = A * np.sin(2 * np.pi * f2 * t_n[n_change:])
    
    # Add noise
    noise = np.random.normal(scale=np.sqrt(noise_var), size=N_total)
    x_n = signal + noise
    
    # True frequency over time
    true_freqs = np.where(t_n < t_change, f1, f2)
    
    # Test different DFT window sizes
    N_values = [1024, 2048]
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_n, true_freqs, label='True frequency $f(t)$', color='blue')
    
    for N_val in N_values:
        timestamps, freqs = freq_detection(x_n, fs, N=N_val)
        plt.plot(timestamps, freqs, label=f'Estimated frequency (N={N_val})')
    
    plt.title("Test 1: Frequency Step Response")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show()


###############################################
# Test 2: Pure Sine with Noise
###############################################
def test_pure_sine_noise():
    """
    Generate a pure sine wave signal and add varying levels of white 
    noise. Evaluate how the average estimated frequency and estimation 
    error change as a function of SNR.
    """
    # Test parameters
    T = 4.0        # Duration of the test signal
    N = int(T * fs)
    f = 885         # True signal frequency
    A = 1.0         # Amplitude of the sine wave
    
    # Generate the clean sine wave
    t_n = np.arange(N) / fs
    clean_signal = A * np.sin(2 * np.pi * f * t_n)
    
    # Define a range of SNR values
    # SNR is varied by changing noise variance
    SNR_values = np.logspace(-2, 2, 20)
    noise_vars = (A**2 / (2 * SNR_values))
    
    avg_estimates = []
    avg_errors = []
    error_bars = []
    
    for noise_var in noise_vars:
        # Add noise
        noise = np.random.normal(0, np.sqrt(noise_var), N)
        x_n = clean_signal + noise
        
        # Run frequency detection
        timestamps, freqs = freq_detection(x_n, fs, N=N_DFT)
        freqs = np.array(freqs)
        
        avg_freq_estimate = np.mean(freqs)
        avg_estimates.append(avg_freq_estimate)
        
        errors = np.abs(freqs - f)
        avg_error = np.mean(errors)
        avg_errors.append(avg_error)
        
        error_bars.append(np.std(errors))
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot average frequency vs. 1/SNR
    inv_SNR = 1 / SNR_values
    
    plt.subplot(2, 1, 1)
    plt.plot(inv_SNR, avg_estimates, label='Average Frequency Estimate', color='red')
    plt.fill_between(inv_SNR, 
                     np.array(avg_estimates) - np.array(error_bars), 
                     np.array(avg_estimates) + np.array(error_bars), 
                     color='red', alpha=0.2)
    plt.axhline(f, color='black', linestyle='--', label=f'True Frequency = {f} Hz')
    plt.xscale('log')
    plt.xlabel('1/SNR')
    plt.ylabel('Frequency Estimate (Hz)')
    plt.title("Test 2: Pure Sine with Noise - Frequency Estimates")
    plt.legend()
    plt.grid(True)
    
    # Plot average error vs. 1/SNR
    plt.subplot(2, 1, 2)
    plt.plot(inv_SNR, avg_errors, label='Average Error', color='blue')
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
def test_real_world_noise(audio_file='./SampleAudio/A4_oboe.wav'):
    """
    Load a real-world audio file (such as a musical instrument note),
    resample to a chosen fs, and add varying noise levels to assess how
    the algorithm performs on complex waveforms under noise.
    """
    # Desired parameters
    target_fs = 16_000   # Choose a suitable sampling rate for analysis
    T = 2.2
    N = int(T * target_fs)
    true_frequency = 440 # Expected fundamental frequency for A4
    
    # Load and preprocess audio
    orig_fs, audio_data = wavfile.read(audio_file)
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))  # Normalize
    
    # Resample if needed
    if orig_fs != target_fs:
        audio_data = resample_poly(audio_data, target_fs, orig_fs)
    
    # Ensure we have N samples (pad or truncate)
    if len(audio_data) < N:
        audio_data = np.pad(audio_data, (0, N - len(audio_data)), 'constant')
    else:
        audio_data = audio_data[:N]
        
    # Calculate signal power
    signal_power = np.mean(audio_data**2)
    
    # Define noise powers and arrays to store results
    noise_powers = np.logspace(-2, 2, 20)
    avg_estimates = []
    avg_errors = []
    variances = []
    SNR_values = []
    
    for noise_power in noise_powers:
        noise = np.random.normal(scale=np.sqrt(noise_power), size=len(audio_data))
        noisy_signal = audio_data + noise
        
        # Frequency detection
        _, freqs = freq_detection(noisy_signal, target_fs, N=2048)
        freqs = np.clip(freqs, 0, target_fs/2)
        
        avg_freq_estimate = np.mean(freqs)
        var = np.var(freqs)
        error = np.abs(avg_freq_estimate - true_frequency)
        
        snr = signal_power / noise_power
        SNR_values.append(snr)
        
        avg_estimates.append(avg_freq_estimate)
        avg_errors.append(error)
        variances.append(var)
    
    inv_SNR_values = 1 / np.array(SNR_values)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(inv_SNR_values, avg_estimates, color='r')
    # Show ±1 std around the mean estimate
    std_devs = np.sqrt(variances)
    ax1.fill_between(inv_SNR_values, 
                     np.array(avg_estimates) - std_devs, 
                     np.array(avg_estimates) + std_devs, 
                     color='r', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_title(f"Test 3: Real-world Signal - Average Frequency Estimate with Noise\n({audio_file})")
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)
    
    ax2.plot(inv_SNR_values, avg_errors, color='b')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("Test 3: Real-world Signal - Estimation Error vs. Noise")
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error |f - f_hat| (Hz)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def test_vocal_noise(audio_file='SampleAudio/Zauberflöte_vocal.wav',
                     target_fs=16000, T=2.2, true_frequency=440, N_DFT=2048):
    # Load and preprocess audio
    orig_fs, audio_data = wavfile.read(audio_file)
    audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))  # Normalize
    
    N = int(T * target_fs)
    
    # Resample if needed
    if orig_fs != target_fs:
        audio_data = resample_poly(audio_data, target_fs, orig_fs)
    
    # Ensure we have N samples (pad or truncate)
    if len(audio_data) < N:
        audio_data = np.pad(audio_data, (0, N - len(audio_data)), 'constant')
    else:
        audio_data = audio_data[:N]
        
    # Calculate signal power
    signal_power = np.mean(audio_data**2)
    
    # Define noise powers and arrays to store results
    noise_powers = np.logspace(-2, 2, 20)
    avg_estimates = []
    avg_errors = []
    variances = []
    SNR_values = []
    
    for noise_power in noise_powers:
        noise = np.random.normal(scale=np.sqrt(noise_power), size=len(audio_data))
        noisy_signal = audio_data + noise
        
        # Frequency detection
        _, freqs = freq_detection(noisy_signal, target_fs, N=N_DFT)
        
        # Clip frequencies to a reasonable range
        freqs = np.clip(freqs, 0, target_fs/2)
        
        avg_freq_estimate = np.mean(freqs)
        var = np.var(freqs)
        error = np.abs(avg_freq_estimate - true_frequency)
        
        snr = signal_power / noise_power
        SNR_values.append(snr)
        
        avg_estimates.append(avg_freq_estimate)
        avg_errors.append(error)
        variances.append(var)
    
    inv_SNR_values = 1 / np.array(SNR_values)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(inv_SNR_values, avg_estimates, color='r')
    std_devs = np.sqrt(variances)
    ax1.fill_between(inv_SNR_values, 
                     np.array(avg_estimates) - std_devs, 
                     np.array(avg_estimates) + std_devs, 
                     color='r', alpha=0.2)
    ax1.set_xscale('log')
    ax1.set_title(f"Vocal Test: Average Frequency Estimate with Noise\n({audio_file})")
    ax1.set_ylabel('Frequency Estimate (Hz)')
    ax1.grid(True)
    
    ax2.plot(inv_SNR_values, avg_errors, color='b')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("Vocal Test: Estimation Error vs. Noise")
    ax2.set_xlabel('1/SNR')
    ax2.set_ylabel('Error |f - f_hat| (Hz)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()  

###############################################
# Run all tests
###############################################
if __name__ == "__main__":
    # Test 1: Step in frequency
    test_frequency_step()
    
    # Test 2: Pure sine wave with noise
    test_pure_sine_noise()
    
    # Test 3: Real-world audio with noise
    test_real_world_noise('./SampleAudio/A4_oboe.wav')

    test_vocal_noise()
