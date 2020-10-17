"""Implements all wavelet transform features.
"""


import pywt
import numpy as np


DEFAULT_WAVELET = 'haar'


def details_coefficients(signal, level='max', wavelet=DEFAULT_WAVELET):
    """Calculates the details coefficients \\(c_D\\) of the given level
    wavelet transform.

    Args:
        signal (numpy.ndarray): (n_samples, window_size)-dimensional numpy array.
        level (int): Level of wavelet transform, defaults to maximum possible.
        wavelet (str): Wavelet to use, defaults to haar (db1).

    Returns:
        numpy.ndarray: Details coefficients as a (n_samples, window_size)-dimensional array.

    """
    if level == 'max':
        level = pywt.dwt_max_level(signal.shape[1], wavelet)

    dwt = pywt.wavedec(signal, wavelet, level=level, axis=1)
    return dwt[1]


def energy(signal, levels=None, wavelet=DEFAULT_WAVELET):
    """Calculates the energy of the wavelet transform for each level.

    Let \\(c_D\\) be the array of details coefficient of a(ny) level of a
    wavelet transform. The energy of that level is then defined as:

    \\[E(c_D) = \\sum_n (c_D[n])^2\\]

    Args:
        signal (numpy.ndarray): (n_samples, window_size)-dimensional numpy array.
        levels (list): List of transform levels, defaults to all possible.
        wavelet (str): Wavelet to use, defaults to haar (db1).

    Returns:
        numpy.ndarray: Energy of (details of) each transform level as a (n_samples, n_levels)-dimensional array.
    """
    # Default to all transform levels possible for given signal
    if levels is None:
        levels = list(range(1, pywt.dwt_max_level(signal.shape[1], wavelet)+1))

    dwt = pywt.wavedec(signal, wavelet, level=max(levels), axis=1)
    dwt = list(reversed(dwt[1:]))   # Otherwise: dwt=[cA_n, cD_n, cD_n-1, ...]

    X = np.empty((signal.shape[0], 0))
    for level in levels:
        new = np.sum(np.square(dwt[level-1]), axis=1).reshape(-1, 1)
        X = np.hstack((X, new))
    return X


def first_level_energy(current, wavelet=DEFAULT_WAVELET):
    """Calculates the energy of the first level wavelet transform.

    This is an ease-of-use interface to the energy() function.

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        wavelet (str): Wavelet to use, defaults to haar (db1).

    Returns:
        numpy.ndarray: Energy as a (n_samples, 1)-dimensional array.
    """
    return energy(current, levels=(1,), wavelet=wavelet)


def all_transform_levels_energy(current, wavelet=DEFAULT_WAVELET):
    """Calculates the energy of the all wavelet transform levels.

    This is an ease-of-use interface to the energy() function.

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        wavelet (str): Wavelet to use, defaults to haar (db1).

    Returns:
        numpy.ndarray: Energy as a (n_samples, max_levels)-dimensional array.
    """
    return energy(current, wavelet=wavelet)


def dominant_scale(signal, wavelet=DEFAULT_WAVELET):
    """Returns the dominant scale (=level) of the wavelet transform.

    The dominant scale is defined as the level with the maximum energy.

    Args:
        signal (numpy.ndarray): (n_samples, window_size)-dimensional numpy array.
        wavelet (str): Wavelet to use, defaults to haar (db1).

    Returns:
        numpy.ndarray: Dominant scale as a (n_samples, 1)-dimensional array.
    """
    dwt_energy = energy(signal, wavelet=wavelet)
    return np.argmax(dwt_energy, axis=1).reshape(-1, 1)


def energy_over_time(signal, n_partitions=5, wavelet=DEFAULT_WAVELET):
    """Calculates the energy over time.

    The first level wavelet transform is applied to given signal and the
    details coefficient \\(c_D\\) then seperated into n_partitions equally
    sized parts \\((c_D(i))_{i \\in \\{1, 2, ..., \\text{n_partitions}\\}}\\).
    Then the energy over time can be calculated as the energy of each
    partition \\((E(c_D(i)))_{i \\in \\{1, 2, ..., \\text{n_partitions}\\}} = 
    (\\sum_n (c_D(i)[n])^2)_{i \\in \\{1, 2, ..., \\text{n_partitions}\\}}\\).

    Args:
        signal (numpy.ndarray): (n_samples, window_size)-dimensional numpy array.
        wavelet (str): Wavelet to use, defaults to haar (db1).

    Returns:
        numpy.ndarray: Energy over time as a (n_samples, n_partitions)-dimensional array.
    """
    if n_partitions == 1:
        return energy(signal, levels=[1])

    dwt = pywt.dwt(signal, wavelet, axis=1)[1]

    part_len = int(np.floor(dwt.shape[1] / n_partitions))
    X = np.empty((signal.shape[0], 0))
    for i in range(n_partitions):
        new = np.sum(np.square(dwt[:, i*part_len:(i+1)*part_len]), axis=1)
        X = np.hstack((X, new.reshape(-1, 1)))

    return X
