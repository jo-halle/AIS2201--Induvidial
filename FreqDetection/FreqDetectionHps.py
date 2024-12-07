import numpy as np

def FreqDetectionHps(xN: np.ndarray, fs: int, N: int = 1024, harmonics: int = 3) -> (np.ndarray, np.ndarray):
    timeStamps = []
    freqs = []
    window = np.hanning(N)  # Optionally use a window here too
    for windowStart in range(0, len(xN), N):
        if windowStart + N > len(xN):
            break
        xSlice = xN[windowStart:windowStart + N]
        xWindowed = xSlice * window
        Xm = np.fft.rfft(xWindowed, n=N)
        Xm[0] = 0
        magnitudeSpectrum = np.abs(Xm)

        # Harmonic Product Spectrum:
        # Downsample the spectrum and multiply
        HPS = magnitudeSpectrum.copy()
        for h in range(2, harmonics + 1):
            # Downsample by factor h
            downsampled = magnitudeSpectrum[::h]
            # Multiply onto HPS (shorten HPS to match downsampled length)
            HPS = HPS[:len(downsampled)] * downsampled

        # Find peak in HPS
        mPeak = np.argmax(HPS)
        freqs.append(mPeak / N * fs)
        timeStamps.append((windowStart + N) / fs)
    return np.array(timeStamps), np.array(freqs)
