import numpy as np
from scipy import fft

from features.helpers import rms, geo_mean, normalize


POWER_FREQUENCY = 50  # Hz
SAMPLING_RATE = 6400  # Measurements / second
PERIOD_LENGTH = SAMPLING_RATE // POWER_FREQUENCY


def _get_default_window(current):
    """Calculates the default window for FFT.

    Window is calculated to be the smallest power of two smaller or equal than
    the window size of the complete current array. Example: input of 2s
    interval = 12800 measurements -> window is calculated to be 8192

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Window as an int.
    """
    return int(2**np.floor(np.log2(current.shape[1])))


def spectrum(current, window=None, sampling_rate=SAMPLING_RATE):
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
    # If window is not specified floor to next power of two
    if window is None:
        window = _get_default_window(current)

    normalized_current = normalize(current[:, :window])
    return np.abs(fft.rfft(normalized_current, axis=1))


def spectral_frequencies(window, n=20, limit_to_harmonics=True,
                         power_frequency=POWER_FREQUENCY,
                         sampling_rate=SAMPLING_RATE):
    """Calculates the spectral frequencies (in Hz) for an application of
    fft.rfft.

    Args:
        window (int): Length of input array.
        limit_to_harmonics (bool): Whether to return harmonic frequencies (True) or the full spectrum.
        n (int): Number of harmonics.

    Returns:
        Spectral frequencies as a list.
    """
    freqs = fft.rfftfreq(window) * sampling_rate
    if limit_to_harmonics:
        return freqs[_get_harmonics_indices(freqs, n=n,
                                            power_frequency=power_frequency)]
    return freqs


def _get_harmonics_indices(spectral_frequencies, n=20,
                           power_freq=POWER_FREQUENCY):
    harmonics = np.arange(power_freq, n * power_freq+1, power_freq)
    freqs = np.where(spectral_frequencies < 0, 0, spectral_frequencies)
    return np.argmin(np.abs(np.dstack([freqs]*n) - harmonics), axis=1).reshape(-1)


def harmonics(current, n=20, window=None,
              power_frequency=POWER_FREQUENCY, sampling_rate=SAMPLING_RATE):
    """Calculates the amplitudes of the first n harmonics frequencies.

    Will use window-sized subsets of current measurements. Note that for
    performance reasons window should be a power of two. By default, window
    will be calculated to be the smallest power of two smaller or equal than
    the window size of the complete current array. Example: input of 2s
    interval = 12800 measurements -> window is calculated to be 8192. You can
    get the corresponding frequencies with the spectral_frequencies method.

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.
        n (int): Number of harmonics.
        window (int): Only use window-sized subset of current measurements.

    Returns:
        Harmonic amplitudes as a (n_samples, n)-dimensional array.
    """
    # If window is not specified floor to next power of two
    if window is None:
        window = _get_default_window(current)

    freqs = spectral_frequencies(window, limit_to_harmonics=False,
                                 power_frequency=power_frequency,
                                 sampling_rate=sampling_rate)
    idxs = _get_harmonics_indices(freqs, n=n, power_frequency=power_frequency)
    return spectrum(current, window)[:, idxs]


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
    odd, even = np.arange(0, 20, step=20), np.arange(1, 20, step=20)
    return (np.mean(harmonics_amp[:, odd], axis=1)
            / np.mean(harmonics_amp[:, even], axis=1)).reshape(-1, 1)


def spectral_flatness(spectrum_amp):
    """Calculates the Spectral flatness (SPF).

    Let \\(x_{f}\\) be the real-part amplitude of the bin with frequency
    \\(f\\) in the current's spectrum. Then:

    \\[SPF = \\frac{\\sqrt[N]{\\prod_{f \\in f_{bins}} x_f}}
                   {\\frac{1}{N} \\sum_{f \\in f_{bins}} x_f}\\]

    Args:
        spectrum_amp: Spectral amplitudes as a (n_samples, window)-dimensional array.

    Returns:
        Spectral flatness as a (n_samples, 1)-dimensional array.
    """
    return (geo_mean(spectrum_amp) / np.mean(spectrum_amp, axis=1)).reshape(-1, 1)


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


def spectral_centroid(spectrum_amp, current, power_frequency=POWER_FREQUENCY,
                      sampling_rate=SAMPLING_RATE):
    """Calculates the Spectral centroid \\(C_f\\).

    Let \\(x_{f}\\) be the real-part amplitude of the bin with frequency
    \\(f\\) in the current's spectrum. Then:

    \\[SPF = \\frac{\\sum_{f \\in f_{bins}} x_f \\times f}
                   {\\sum_{f \\in f_{bins}} x_f}\\]

    Args:
        spectrum_amp: Spectral amplitudes as a (n_samples, window)-dimensional array.

    Returns:
        Spectral centroid as a (n_samples, 1)-dimensional array.
    """
    window = _get_default_window(current)
    freqs = spectral_frequencies(window, limit_to_harmonics=False,
                                 power_frequency=power_frequency,
                                 sampling_rate=sampling_rate)[1:]

    return (np.sum(spectrum_amp[:, 1:] / freqs, axis=1)
            / np.sum(spectrum_amp[:, 1:], axis=1)).reshape(-1, 1)


def harmonic_spectral_centroid(current, power_frequency=POWER_FREQUENCY,
                               sampling_rate=SAMPLING_RATE):
    """Calculates the spectral centroid \\(C_h\\).

    \\[C_h = \\frac{\\sum_{i=1}^{50} x_{f_i} \\cdot i}
                   {\\sum_{i=1}^{50} x_{f_i}}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Harmonic spectral centroid as a (n_samples, 1)-dimensional array.
    """
    harmonics_amp = harmonics(current, n=50, power_frequency=power_frequency,
                              sampling_rate=sampling_rate)
    return (np.sum(harmonics_amp * np.arange(1, 51), axis=1)
            / np.sum(harmonics_amp, axis=1)).reshape(-1, 1)


def signal_to_signal_mean_ratio(spectrum_amp):
    """Calculates the signal to signal mean ratio (SSMR).

    Let \\(x_{f}\\) be the real-part amplitude of the bin with frequency
    \\(f\\) in the current's spectrum. Then:

    \\[SSMR = \\frac{\\max_{f \\in f_{bins}} x_f}
                    {\\frac{1}{N} \\sum_{f \\in f_{bins}} x_f}\\]

    Args:
        spectrum_amp: Spectral amplitudes as a (n_samples, window)-dimensional array.

    Returns:
        Signal to signal mean ratio as a (n_samples, 1)-dimensional array.
    """
    return (np.max(spectrum_amp, axis=1)
            / np.mean(spectrum_amp, axis=1)).reshape(-1, 1)


def second_harmonic(harmonics_amp):
    """Calculates the amplitude of the second harmonic.

    You can get the corresponding frequency with the spectral_frequencies
    method (use the second element of its return value).

    Args:
        harmonics_amp: Harmonic amplitudes as a (n_samples, n)-dimensional array.

    Returns:
        Amplitude of the second harmonic as a (n_samples, 1)-dimensional array.
    """
    return harmonics_amp[:, 1].reshape(-1, 1)
