import math
import numpy as np
from scipy import fft
from scipy import signal

from features import helpers
from features.helpers import rms, average_periods, normalize


POWER_FREQUENCY = 50  # Hz
SAMPLING_RATE = 6400  # Measurements / second
PERIOD_LENGTH = SAMPLING_RATE // POWER_FREQUENCY


def phase_shift(voltage, current, mains_frequency=POWER_FREQUENCY,
                power_frequency=POWER_FREQUENCY, sampling_rate=SAMPLING_RATE):
    """Calculates Phase shift (unit: radian).

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Phase shift as a (n_samples, 1)-dimensional array.
    """
    # Avoid circular imports
    from features.spectral import _get_default_window
    n = _get_default_window(current)
    freqs = fft.rfftfreq(n) * sampling_rate
    # Get only the f0 bin of the spectrum
    idx = np.argmin(np.abs(freqs - mains_frequency))
    spec_current = fft.rfft(current, n, axis=1)[:, idx]
    spec_voltage = fft.rfft(voltage, n, axis=1)[:, idx]
    # Phase shift = phase angle (f0 bin) current - phase angle voltage
    return (np.angle(spec_current) - np.angle(spec_voltage)).reshape(-1, 1)


def active_power(voltage, current, phase_shift=None, period_length=PERIOD_LENGTH):
    """Calculates Active power (P).

    If phase shift is None, it is calculated.

    \\[P = \\text{rms}(V) \\times \\text{rms}(I) \\times \\cos(\\phi)\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        phase_shift (numpy.ndarray): (n_samples, 1)-dimensional array of phase shifts.

    Returns:
        numpy.ndarray: Active power as a (n_samples, 1)-dimensional array.
    """
    if phase_shift is None:
        phase_shift = phase_shift(voltage, current,
                                  period_length=period_length)

    return rms(voltage) * rms(current) * np.cos(phase_shift)


def reactive_power(voltage, current, phase_shift=None, period_length=PERIOD_LENGTH):
    """Calculates Reactive power (Q).

    If phase shift is None, it is calculated.

    \\[Q = \\text{rms}(V) \\times \\text{rms}(I) \\times \\sin(\\phi)\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        phase_shift (numpy.ndarray): (n_samples, 1)-dimensional array of phase shifts.

    Returns:
        numpy.ndarray: Reactive power as a (n_samples, 1)-dimensional array.
    """
    if phase_shift is None:
        phase_shift = phase_shift(voltage, current,
                                  period_length=period_length)

    return rms(voltage) * rms(current) * np.sin(phase_shift)


def apparent_power(voltage, current):
    """Calculates Apparent power (S).

    \\[S = \\text{rms}(V) \\times \\text{rms}(I)\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Apparent power as a (n_samples, 1)-dimensional array.
    """
    return rms(voltage) * rms(current)


def vi_trajectory(voltage, current, num_samples=20, num_periods=10,
                  normalize=True, period_length=PERIOD_LENGTH):
    """Calculates the VI-trajectory.

    Averages the first num_periods periods of measurements, extracts
    n_samples equidistant sampling points, and normalizes by dividing by
    maximum.

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        num_samples (int): Number of sampling points used.
        num_periods (int): Number of periods used for sampling
        normalize (bool): Whether to normalize (divide by max).

    Returns:
        numpy.ndarray: VI-trajectory as tuple of (n_samples, num_samples)-dimensional arrays of normalized voltage and current values.
    """
    # Calculate equidistant sampling points within each period
    sample = np.linspace(0, period_length-1, num=num_samples,
                         endpoint=False, dtype='int')

    # Calculate mean of each sampling point over num_periods periods
    v = voltage[:, :num_periods*period_length]
    i = current[:, :num_periods*period_length]
    v = v.reshape(-1, num_periods, period_length)[:, :, sample]
    i = i.reshape(-1, num_periods, period_length)[:, :, sample]
    v = np.mean(v, axis=1)
    i = np.mean(i, axis=1)

    # Normalize voltage and current to [0,1] by dividing by range
    if normalize:
        v = v / np.max(np.abs(v)).reshape(-1, 1)
        i = i / np.max(np.abs(i)).reshape(-1, 1)
    return v, i


def form_factor(current):
    """Calculates Form factor (FF).

    \\[FF = \\frac{\\text{rms}(I)}{\\text{mean}(|I|)}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Form factor as a (n_samples, 1)-dimensional array.
    """
    return rms(current) / np.mean(np.abs(current), axis=1).reshape(-1, 1)


def crest_factor(current):
    """Calculates Crest factor (CF).

    \\[CF = \\frac{\\max(|I|)}{\\text{rms}(I)}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Crest factor as a (n_samples, 1)-dimensional array.
    """
    return np.max(np.abs(current), axis=1).reshape(-1, 1) / rms(current)


def resistance_mean(voltage, current):
    """Calculates Resistance \\(R_\\text{mean}\\) (mean version) from
    voltage and current arrays.

    \\[R_\\text{mean} = \\frac{\\sqrt{\\text{mean}(V^2)}}{\\sqrt{\\text{mean}(I^2)}}\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Resistance as a (n_samples, 1)-dimensional array.
    """
    return rms(voltage) / rms(current)


def resistance_median(voltage, current):
    """Calculates Resistance \\(R_\\text{median}\\) (median version) from
    voltage and current arrays.

    \\[R_\\text{median} = \\frac{\\sqrt{\\text{median}(V^2)}}{\\sqrt{\\text{median}(I^2)}}\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Resistance as a (n_samples, 1)-dimensional array.
    """
    numerator = np.sqrt(np.median(np.square(voltage), axis=1))
    denominator = np.sqrt(np.median(np.square(current), axis=1))
    return (numerator / denominator).reshape(-1, 1)


def admittance_mean(voltage, current):
    """Calculates Admittance \\(Y_\\text{mean}\\) (mean version) from
    voltage and current arrays.

    \\[Y_\\text{mean} = \\frac{1}{R_\\text{mean}} =
    \\frac{\\sqrt{\\text{mean}(I^2)}}{\\sqrt{\\text{mean}(V^2)}}\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Admittance as a (n_samples, 1)-dimensional array.
    """
    return 1 / resistance_mean(voltage, current)


def admittance_median(voltage, current):
    """Calculates Admittance \\(Y_\\text{median}\\) (median version) from
    voltage and current arrays.

    \\[Y_\\text{median} = \\frac{1}{R_\\text{median}} =
    \\frac{\\sqrt{\\text{median}(I^2)}}{\\sqrt{\\text{median}(V^2)}}\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Admittance as a (n_samples, 1)-dimensional array.
    """
    return 1 / resistance_median(voltage, current)


def log_attack_time(current, sampling_rate=SAMPLING_RATE):
    """Calculates the LogAttackTime \\(\\ln(\\underset{t}{\\arg \\max}(I_t))\\).

    Unit used: ms.

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: LogAttackTime as a (n_samples, 1)-dimensional array.
    """
    starting_times = np.argmax(current, axis=1).reshape(-1, 1)
    starting_times = np.where(starting_times > 0, starting_times, 1)

    # Make sure input to log is > 0
    return np.log(starting_times / (sampling_rate / 1000))


def temporal_centroid(current, mains_frequency=POWER_FREQUENCY,
                      period_length=PERIOD_LENGTH):
    """Calculates the Temporal centroid \\(C_t\\).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements. Then \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\) and
    \\[C_t = \\frac{1}{f_0} \\cdot \\frac{\\sum_{k=1}^N I_{P(k)} \\cdot k}
       {\\sum_{k=1}^N I_{P(k)}} \\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        mains_frequency (int): Mains frequency, defaults to power frequency.

    Returns:
        numpy.ndarray: Temporal centroid as a (n_samples, 1)-dimensional array.
    """
    # Calculate RMS of each period
    # TODO: Use apply_to_periods?
    ip = rms(current.reshape(current.shape[0], -1, period_length), axis=2)

    # Calculate numerator and denominator and put together
    numerator = np.sum(ip * np.arange(1, ip.shape[1]+1), axis=1)
    denominator = np.sum(ip, axis=1)
    return mains_frequency * (numerator / denominator).reshape(-1, 1)


def inrush_current_ratio(current, period_length=PERIOD_LENGTH):
    """Calculates the Inrush current ratio (ICR).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements. Then \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\) and

    \\[ICR =  \\frac{I_{P(1)}}{I_{P(N)}}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Inrush current ratio as a (n_samples, 1)-dimensional array.
    """
    return rms(current[:, :period_length]) / rms(current[:, -period_length:])


def positive_negative_half_cycle_ratio(current, period_length=PERIOD_LENGTH):
    """Calculates Positive-negative half cycle ratio (PNR).

    Let \\(I_{P_\\text{pos}}\\) and \\(I_{P_\\text{neg}}\\) be the RMS of 10
    averaged positive and negative current half cycles. Then

    \\[PNR = \\frac{\\min\\{I_{P_\\text{pos}}, I_{P_\\text{neg}}\\}}
                   {\\max\\{I_{P_\\text{pos}}, I_{P_\\text{neg}}\\}}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: PNR as a (n_samples, 1)-dimensional array.
    """
    # Create indices for positive and negative half cycles of sin wave
    idcs = np.arange(0, period_length * 10)
    p_idcs = idcs[idcs % period_length < period_length / 2]
    n_idcs = idcs[idcs % period_length >= period_length / 2]

    p_current = current[:, p_idcs].reshape(-1, 10, period_length // 2)
    n_current = current[:, n_idcs].reshape(-1, 10, period_length // 2) * -1

    # Calculate RMS of averaged half cycles
    p_n = np.hstack((rms(np.mean(p_current, axis=1)),
                     rms(np.mean(n_current, axis=1))))

    return (np.min(p_n, axis=1) / np.max(p_n, axis=1)).reshape(-1, 1)


def max_min_ratio(current):
    """Calculates the Max-min ratio (MAMI).

    \\[MAMI = \\frac{\\min\\{|\\max(I)|, |\\min(I)|\\}}
                    {\\max\\{|\\max(I)|, |\\min(I)|\\}}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Max-min ratio as a (n_samples, 1)-dimensional array.
    """
    extrema = np.hstack((np.abs(np.max(current, axis=1)).reshape(-1, 1),
                         np.abs(np.min(current, axis=1)).reshape(-1, 1)))
    return (np.min(extrema, axis=1) / np.max(extrema, axis=1)).reshape(-1, 1)


def peak_mean_ratio(current):
    """Calculates the Peak-mean ratio (PMR).

    \\[PMR = \\frac{\\max(|I|)}{\\text{mean}(|I|)}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Peak-mean ratio as a (n_samples, 1)-dimensional array.
    """
    return (np.max(np.abs(current), axis=1)
            / np.mean(np.abs(current), axis=1)).reshape(-1, 1)


def max_inrush_ratio(current, period_length=PERIOD_LENGTH):
    """Calculates the Max inrush ratio (MIR).

    Let \\(I_{W(k)}\\) be the current measurements of the \\(k\\)th period and
    \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\). Then

    \\[MIR = \\frac{I_{P(1)}}{\\max(|I_{W(1)}|)}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Max inrush ratio as a (n_samples, 1)-dimensional array.
    """
    first_period_rms = rms(current[:, :period_length])
    first_period_max = np.max(np.abs(current[:, :period_length]), axis=1)
    return first_period_rms / first_period_max.reshape(-1, 1)


def mean_variance_ratio(current):
    """Calculates the Mean variance ratio (MVR).

    \\[MVR = \\frac{\\text{mean}(|I|)}
                   {\\text{var}(|I|)}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Mean variance ratio as a (n_samples, 1)-dimensional array.
    """
    return (np.mean(np.abs(current), axis=1)
            / np.var(np.abs(current), axis=1)).reshape(-1, 1)


def waveform_distortion(current, period_length=PERIOD_LENGTH):
    """Calculates the Waveform distortion WFD.

    Let \\(I_{W_\\text{avg}}\\) be the current measurements (aligned with the
    rising zero crossing) averaged over the first ten periods and normalized
    (with their maximum value). Let \\(Y_{\\sin} = \\left(\\sin(\\frac{i}{128}
    \\times 2 \\pi)\\right)_{i \\in (0, 1, ..., 127)}\\). Then:

    \\[WFD = \\sum \\left( |Y_{\\sin}| - |I_{W_\\text{avg}}| \\right)\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Waveform distortion as a (n_samples, 1)-dimensional array.
    """
    current = normalize(average_periods(current, 10, period_length),
                        method='max')
    y = np.sin(np.linspace(0, 2*np.pi, period_length))
    return np.sum(np.abs(current) - np.abs(y), axis=1).reshape(-1, 1)


def waveform_approximation(current, period_length=PERIOD_LENGTH):
    """Calculates the Waveform approximation (WFA).

    Let \\(I_{W_\\text{avg}}\\) be the current measurements (aligned with the
    rising zero crossing) averaged over the first ten periods and normalized
    (with their maximum value). Let \\(S = (0, 6, 12, 19, ..., 114, 120)\\) be
    a list of 20 equidistant sampling points over one period. Then:

    \\[WFA = \\left( I_{W_\\text{avg}, s} \\right)_{s \\in S}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Waveform approximation as a (n_samples, 20)-dimensional array.
    """
    current = normalize(average_periods(current, 10, period_length),
                        method='max')
    sampling_points = np.linspace(0, period_length-1, num=20,
                                  endpoint=False, dtype='int')
    return current[:, sampling_points]


def current_over_time(current, period_length=PERIOD_LENGTH):
    """Calculates the Current over time (COT).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements. Then \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\) and

    \\[COT = \\left( I_{P(1)}, I_{P(2)}, ..., I_{P(25)} \\right)\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Current over time as a (n_samples, 25)-dimensional array.
    """
    return helpers.apply_to_periods(current, rms, 25, (0),
                                    period_length=period_length)


def admittance_over_time(voltage, current, period_length=PERIOD_LENGTH):
    """Calculates the Admittance over time (AOT).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements. Then \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\). Further, let
    \\(V_{W(k)}\\) and \\(V_{P(k)}\\) be analogous for the voltage.

    \\[AOT = \\left( \\frac{I_{P(1)}}{V_{P(1)}}, \\frac{I_{P(2)}}{V_{P(2)}},
    ..., \\frac{I_{P(25)}}{V_{P(25)}} \\right)\\]

    Args:
        voltage (numpy.ndarray): (n_samples, window_size)-dimensional array of voltage measurements.
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Admittance over time as a (n_samples, 25)-dimensional array.
    """
    return (helpers.apply_to_periods(current, rms, 25, (0),
                                     period_length=period_length) /
            helpers.apply_to_periods(voltage, rms, 25, (0),
                                     period_length=period_length))


def periods_to_steady_state_current(current, period_length=PERIOD_LENGTH):
    """Calculates the Periods to steady state current (PSS).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements. Then \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\) and

    \\[PSS = \\underset{k}{\\arg \\min} \\thinspace I_{P(k)} < \\text{median}
    (COT) + \\frac{1}{8} \\cdot (\\max(COT) - \\text{median}(COT))\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        numpy.ndarray: Periods to steady state current as a (n_samples, 1)-dimensional array.
    """
    cot = current_over_time(current, period_length=period_length)
    l = (1/8 * (np.max(cot, axis=1) - np.median(cot, axis=1)) +
         np.median(cot, axis=1))
    return np.argmax(cot > l.reshape(-1, 1), axis=1).reshape(-1, 1) + 1


def transient_steady_states_ratio(current, n_periods=5,
                                  period_length=PERIOD_LENGTH):
    """Calculates the Transient steady states ratio (TSSR).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements and \\(\\langle t_1, t_2, ..., t_n \\rangle\\) the
    concatenation of tuples \\(t_1, ..., t_n\\). Then:

    \\[TSSR = \\frac{\\text{rms}(\\langle I_{W(1)}, I_{W(2)}, ...,
    I_{W(\\text{n_periods})}\\rangle)}
    {\\text{rms}(\\langle I_{W(N - \\text{n_periods})},
    I_{W(N - \\text{n_periods} + 1)}, ..., I_{W(N)}\\rangle)}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        n_periods (int): Length of transient and steady state in periods

    Returns:
        numpy.ndarray: Transient steady states ratio as a (n_samples, 1)-dimensional array.
    """
    return (rms(current[:, :n_periods*period_length])
            / rms(current[:, -n_periods*period_length:])).reshape(-1, 1)


def current_rms(current, period_length=PERIOD_LENGTH):
    """Calculates the Current RMS (CRMS).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements, \\(\\langle t_1, t_2, ..., t_n \\rangle\\) the
    concatenation of tuples \\(t_1, ..., t_n\\), and \\(\\nu = N \\mod 10\\).
    Then:

    \\[CRMS = \\begin{pmatrix}\\text{rms}(\\langle I_{W(1)}, I_{W(2)}, ...,
    I_{W(10)}\\rangle) \\\\ \\text{rms}(\\langle I_{W(11)}, I_{W(12)}, ...,
    I_{W(20)}\\rangle) \\\\ ... \\\\ \\text{rms}(\\langle I_{W(N - \\nu + 1)},
    I_{W(N - \\nu + 2)}, ..., I_{W(N)}\\rangle)\\end{pmatrix}\\]

    Args:
        current (numpy.ndarray): (n_samples, window_size)-dimensional array of current measurements.
        n_periods (int): Length of transient and steady state in periods

    Returns:
        numpy.ndarray: Transient steady states ratio as a (n_samples, 1)-dimensional array.
    """
    n = int(math.floor(current.shape[1] / period_length / 10))
    cutoff = n*10*period_length
    first = rms(current[:, :cutoff].reshape(-1, n, 10*period_length), axis=2)

    # Handle window_size is not multiple of 10*period_length
    if cutoff != current.shape[1]:
        return np.hstack((first, rms(current[:, cutoff:])))
    return first
