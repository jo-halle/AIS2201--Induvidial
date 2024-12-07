import numpy as np

def FreqDetectionWindowed(xN: np.ndarray, fs: int, N: int = 1024) -> (np.ndarray, np.ndarray):
    timeStamps = []
    freqs = []
    window = np.hanning(N)
    for windowStart in range(0, len(xN), N):
        if windowStart + N > len(xN):
            break
        xSlice = xN[windowStart:windowStart + N]
        xWindowed = xSlice * window
        Xm = np.fft.rfft(xWindowed, n=N)
        Xm[0] = 0
        mPeak = np.argmax(np.abs(Xm))
        freqs.append(mPeak / N * fs)
        timeStamps.append((windowStart + N) / fs)
    return np.array(timeStamps), np.array(freqs)
