import numpy as np

def freqDetectionZeroPad(xN: np.ndarray, fs: int, N: int = 1024, padFactor: int = 4) -> (np.ndarray, np.ndarray):
    timeStamps = []
    freqs = []
    nPadded = N * padFactor
    for windowStart in range(0, len(xN), N):
        if windowStart + N > len(xN):
            break
        xSlice = xN[windowStart:windowStart + N]

        # Zero-pad to nPadded
        xPadded = np.pad(xSlice, (0, nPadded - N), 'constant')
        Xm = np.fft.rfft(xPadded, n=nPadded)
        Xm[0] = 0
        mPeak = np.argmax(np.abs(Xm))
        freqs.append(mPeak / nPadded * fs)
        timeStamps.append((windowStart + N) / fs)
    return np.array(timeStamps), np.array(freqs)
