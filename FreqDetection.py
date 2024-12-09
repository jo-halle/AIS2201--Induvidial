import numpy as np

def FreqDetectionZeroPad(xN: np.ndarray, fs: int, N: int = 1024, padFactor: int = 4) -> (np.ndarray, np.ndarray):
    timeStamps = []
    freqs = []
    nPadded = N * padFactor
    for windowStart in range(0, len(xN), N):
        if windowStart + N > len(xN):
            break
        xSlice = xN[windowStart:windowStart + N]

        xPadded = np.pad(xSlice, (0, nPadded - N), 'constant')
        Xm = np.fft.rfft(xPadded, n=nPadded)
        Xm[0] = 0
        mPeak = np.argmax(np.abs(Xm))
        freqs.append(mPeak / nPadded * fs)
        timeStamps.append((windowStart + N) / fs)
    return np.array(timeStamps), np.array(freqs)


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


def FreqDetectionHps(xN: np.ndarray, fs: int, N: int = 1024, harmonics: int = 3) -> (np.ndarray, np.ndarray):
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
        magnitudeSpectrum = np.abs(Xm)

        
        HPS = magnitudeSpectrum.copy()
        for h in range(2, harmonics + 1):
            downsampled = magnitudeSpectrum[::h]
            HPS = HPS[:len(downsampled)] * downsampled

        mPeak = np.argmax(HPS)
        freqs.append(mPeak / N * fs)
        timeStamps.append((windowStart + N) / fs)
    return np.array(timeStamps), np.array(freqs)


def CombinedFreqDetection(XN, Fs, N=1024, padFactor=4, harmonics=3):
    T_zp, F_zp = FreqDetectionZeroPad(XN, Fs, N=N, padFactor=padFactor)

    T_w, F_w = FreqDetectionWindowed(XN, Fs, N=N)

    T_h, F_h = FreqDetectionHps(XN, Fs, N=N, harmonics=harmonics)

    F_combined = (F_zp + F_w + F_h) / 3

    return T_zp, F_combined
