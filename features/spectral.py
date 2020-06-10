import numpy as np
from scipy import fftpack

from features.helpers import rms, geo_mean


SAMPLING_RATE = 6400


def spectral_frequencies(window, n=20, limit_to_harmonics=True):
    """Calculates the spectral frequencies (in Hz) for an application of
    fftpack.fft.

    Args:
        window (int): Length of input array.
        limit_to_harmonics (bool): Whether to return harmonic frequencies (True) or the full spectrum.
        n (int): Number of harmonics.

    Returns:
        Spectral frequencies as a list.
    """
    freqs = fftpack.fftfreq(window) * SAMPLING_RATE
    if limit_to_harmonics:
        idcs = (freqs % 50 == 0) & (freqs > 0) & (freqs <= n*50)
        return freqs[idcs]
    return freqs


def _harmonics_single(current, n=20, window=None):
    """Calculates the amplitudes of the first n harmonics frequencies.

    Args:
        current: (window_size,)-dimensional array of current measurements.
        n (int): Number of harmonics.
        window (int): Only use window-sized subset of current measurements.

    Returns:
        Harmonic amplitudes as a (n,)-dimensional array.
    """
    # If window is not specified floor to next power of two
    if window is None:
        window = int(2**np.floor(np.log2(current.shape[0])))

    # Normalize input current
    normalized_current = current[:window] / np.max(current[:window])

    # Calculate FFT and corresponding frequencies (in Hz)
    X = fftpack.fft(normalized_current)
    freqs = spectral_frequencies(window, limit_to_harmonics=False)

    # Limit spectrum to harmonics
    harmonics_idcs = (freqs % 50 == 0) & (freqs > 0) & (freqs <= n*50)
    return np.abs(X[harmonics_idcs])


def harmonics(current, n=20, window=None):
    """Calculates the amplitudes of the first n harmonics frequencies.

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.
        n (int): Number of harmonics.
        window (int): Only use window-sized subset of current measurements.

    Returns:
        Harmonic amplitudes as a (n_samples, n)-dimensional array.
    """
    return np.apply_along_axis(_harmonics_single, 1, current, n=n, window=window)


def _spectrum_single(current, window=None):
    """Calculates the spectrum (absolute amplitude for positive frequencies)
    for a single (one-dimensional) current array.

    Will use window-sized subsets of current measurements. Note that for
    performance reasons window should be a power of two. By default, window
    will be calculated to be the smallest power of two smaller or equal than
    the window size of the complete current array. Example: input of 2s
    interval = 12800 measurements -> window is calculated to be 8192. You can
    get the corresponding frequencies with the spectral_frequencies method.

    Args:
        current: (window_size,)-dimensional array of current measurements.
        window: Optional, use window-sized subsets of current measurements for FFT.

    Returns:
        Spectrum as a (n_frequencies,)-dimensional array.
    """
    # If window is not specified floor to next power of two
    if window is None:
        window = int(2**np.floor(np.log2(current.shape[0])))

    # Normalize input current
    normalized_current = current[:window] / np.max(current[:window])

    # Calculate FFT and corresponding frequencies (in Hz)
    X = fftpack.fft(normalized_current)
    freqs = spectral_frequencies(window, limit_to_harmonics=False)

    # Limit spectrum to harmonics
    spectrum_idcs = (freqs > 0) & (freqs <= SAMPLING_RATE / 2)
    return np.abs(X[spectrum_idcs])


def spectrum(current, window=None):
    """Calculates the spectrum (absolute amplitude for positive frequencies)
    for (two-dimensional) arrays of current measurements.

    Will use window-sized subsets of current measurements. Note that for
    performance reasons window should be a power of two. By default, window
    will be calculated to be the smallest power of two smaller or equal than
    the window size of the complete current array. Example: input of 2s
    interval = 12800 measurements -> window is calculated to be 8192. You can
    get the corresponding frequencies with the spectral_frequencies method.

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.
        window: Optional, use window-sized subsets of current measurements for FFT.

    Returns:
        Spectrum as a (n_samples, n_frequencies)-dimensional array.
    """
    return np.apply_along_axis(_spectrum_single, 1, current)


def odd_even_ratio(harmonics_amp):
    """Calculates Odd-even ratio (OER).

    Let \\(x_{f_1}, ..., x_{f_{20}}\\) be the amplitudes of the first 20
    harmonics of the current. Then:

    \\[OER = \\frac{\\text{mean}(x_{f_1}, x_{f_3}, ..., x_{f_{19}})}
                   {\\text{mean}(x_{f_2}, x_{f_4}, ..., x_{f_{20}})}\\]

    Args:
        harmonics_amp: Harmonic amplitudes as a (n_samples, n)-dimensional array.

    Returns:
        Odd-even ratio as a (n_samples, 1)-dimensional array.
    """
    odd = np.mean(harmonics_amp[:, np.arange(0, 20, step=2)], axis=1)
    even = np.mean(harmonics_amp[:, np.arange(1, 20, step=2)], axis=1)
    return (odd / even).reshape(-1, 1)


def spectral_flatness(harmonics_amp):
    """Calculates the Spectral flatness (SPF).

    TODO: Explain bins & x_f

    Let \\(x_{f_1}, ..., x_{f_{20}}\\) be the amplitudes of the first 20
    harmonics of the current. Then:

    \\[SPF = \\frac{\\sqrt[N]{\\prod_{f \\in f_{bins}} x_f}}
                   {\\frac{1}{N} \\sum_{f \\in f_{bins}} x_f}
           \\overset{\\propto}{\\sim}
           \\frac{\\sqrt[20]{\\prod_{i=1}^{20} x_{f_i}}}
                 {\\frac{1}{20} \\sum_{i=1}^{20} x_{f_i}}\\]

    Args:
        harmonics_amp: Harmonic amplitudes as a (n_samples, n)-dimensional array.

    Returns:
        Spectral flatness as a (n_samples, 1)-dimensional array.
    """
    return (geo_mean(harmonics_amp) / np.mean(harmonics_amp, axis=1)).reshape(-1, 1)


def harmonics_energy_distribution(harmonics_amp):
    """Calculates the Harmonics energy distribution (HED).

    Let \\(x_{f_1}, ..., x_{f_{20}}\\) be the amplitudes of the first 20
    harmonics of the current. Then:

    \\[HED = \\frac{1}{x_{f_1}} \\times [x_{f_2}, x_{f_3}, ..., x_{f_{20}}]\\]

    Args:
        harmonics_amp: Harmonic amplitudes as a (n_samples, n)-dimensional array.

    Returns:
        Harmonics energy distribution as a (n_samples, 19)-dimensional array.
    """
    return harmonics_amp[:, 1:] / harmonics_amp[:, 0].reshape(-1, 1)


def tristiumulus(harmonics_amp):
    """Calculates the Tristimulus.

    Let \\(x_{f_1}, ..., x_{f_{20}}\\) be the amplitudes of the first 20
    harmonics of the current. Then the tristimulus is the triple containing:

    \\[T_1 = \\frac{x_{f_1}}{\\sum_{i=1}^{20} x_{f_i}} \\quad \\quad
       T_2 = \\frac{x_{f_2} + x_{f_3} + x_{f_4}}{\\sum_{i=1}^{20} x_{f_i}}\\]
    \\[T_3 = \\frac{x_{f_5} + x_{f_6} + ... + x_{f_{10}}}{\\sum_{i=1}^{20} x_{f_i}}\\]

    Args:
        harmonics_amp: Harmonic amplitudes as a (n_samples, n)-dimensional array.

    Returns:
        Tristimulus as a (n_samples, 3)-dimensional array.
    """
    harmonics_sum = np.sum(harmonics_amp, axis=1)
    t_1 = harmonics_amp[:, 0] / harmonics_sum
    t_2 = np.sum(harmonics_amp[:, (1, 2, 3)], axis=1) / harmonics_sum
    t_3 = np.sum(harmonics_amp[:, 4:9], axis=1) / harmonics_sum
    return np.hstack((t_1.reshape(-1, 1), t_2.reshape(-1, 1),
                      t_3.reshape(-1, 1)))


def total_harmonic_distortion(harmonics_amp):
    """Calculates the Total harmonic distortion (THD).

    Let \\(x_{f_1}, ..., x_{f_{20}}\\) be the amplitudes of the first 20
    harmonics of the current. Then:

    \\[THD = \\frac{\\text{rms}([x_{f_2}, x_{f_3}, ..., x_{f_{20}}])}
                   {x_{f_1}}\\]

    Args:
        harmonics_amp: Harmonic amplitudes as a (n_samples, n)-dimensional array.

    Returns:
        Total harmonic distortion as a (n_samples, 1)-dimensional array.
    """
    # TODO: Check definition with Daniel
    return (rms(harmonics_amp[:, 1:]) / harmonics_amp[:, 0].reshape(-1, 1))


def spectral_centroid(harmonics_amp, current):
    """Calculates the Spectral centroid \\(C_f\\).

    TODO: Explain bins & x_f

    Let \\(x_{f_1}, ..., x_{f_{20}}\\) be the amplitudes of the first 20
    harmonics of the current. Then:

    \\[SPF = \\frac{\\sum_{f \\in f_{bins}} x_f \\times f}
                   {\\sum_{f \\in f_{bins}} x_f}
           \\overset{\\propto}{\\sim}
           \\frac{\\sum_{i=0}^{20} x_{f_i} \\times f_i}
                 {\\sum_{i=0}^{20} x_{f_i}}\\]

    Args:
        harmonics_amp: Harmonic amplitudes as a (n_samples, n)-dimensional array.

    Returns:
        Spectral centroid as a (n_samples, 1)-dimensional array.
    """
    freqs = 1 / spectral_frequencies(current.shape[1], 20)
    numerator = np.sum(harmonics_amp * freqs, axis=1).reshape(-1, 1)
    denominator = np.sum(harmonics_amp, axis=1).reshape(-1, 1)
    return numerator / denominator


def harmonic_spectral_centroid(current):
    """Calculates the spectral centroid \\(C_h\\).

    \\[C_h = \\frac{\\sum_{i=1}^{50} x_{f_i} \\cdot i}
                   {\\sum_{i=1}^{50} x_{f_i}}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Harmonic spectral centroid as a (n_samples, 1)-dimensional array.
    """
    harmonics_amp = harmonics(current, n=50)
    numerator = np.sum(harmonics_amp * np.arange(1, 51), axis=1)
    denominator = np.sum(harmonics_amp, axis=1)
    return (numerator / denominator).reshape(-1, 1)


def signal_to_signal_mean_ratio(current):
    """Calculates the signal to signal mean ratio (SSMR).

    \\[SSMR = \\frac{\\max(\\text{spec})}{\\text{mean}(\\text{spec})}\\] where
    \\(\\text{spec}\\) is the absolute amplitude of positive frequencies in
    the spectrum.

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Signal to signal mean ratio as a (n_samples, 1)-dimensional array.
    """
    spec = np.apply_along_axis(_spectrum_single, 1, current)
    return (np.max(spec, axis=1) / np.mean(spec, axis=1)).reshape(-1, 1)


##############################################################################
#                                                                            #
#                        START EXPERIMENTAL FEATURES                         #
#                                                                            #
##############################################################################


def second_harmonic(harmonics_amp):
    # Second harmonic as one-dimensional feature
    return harmonics_amp[:, 1].reshape(-1, 1)
