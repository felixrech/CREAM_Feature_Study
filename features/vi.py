import numpy as np
from scipy import signal

from features import helpers
from features.helpers import rms, average_periods, normalize


PERIOD_LENGTH = 6400 // 50      # Each period is 128 sampling points


def _phase_shift_single(voltage, current, radian=True):
    """Calculates phase shift of a single sample.

    Args:
        voltage: (window_size, )-dimensional array of voltage measurements.
        current: (window_size, )-dimensional array of current measurements.
        radian (bool): Whether to return as radian (True) or absolute value.

    Returns:
        Phase shift as int.
    """
    # Correlate voltage and current
    corr = signal.correlate(voltage, current)
    corr = corr[corr.size//2:corr.size//2+PERIOD_LENGTH]
    # Maximize correlation (as number of measurements that v leads i)
    phase_shift = corr.argmax() % PERIOD_LENGTH
    phase_shift = phase_shift if phase_shift == 0 else 128 - phase_shift
    if radian:
        return (phase_shift / 128) * 2 * np.pi
    return phase_shift


def phase_shift(voltage, current):
    """Calculates Phase shift.

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Phase shift as a (n_samples, 1)-dimensional array.
    """
    ps = [_phase_shift_single(v, i) for v, i in zip(voltage, current)]
    return np.array(ps).reshape(-1, 1)


def active_power(voltage, current, phase_shift):
    """Calculates Active power (P).

    \\[P = \\text{rms}(V) \\times \\text{rms}(I) \\times \\cos(\\phi)\\]

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.
        phase_shift: (n_samples, 1)-dimensional array of phase shifts.

    Returns:
        Active power as a (n_samples, 1)-dimensional array.
    """
    return rms(voltage) * rms(current) * np.cos(phase_shift)


def reactive_power(voltage, current, phase_shift):
    """Calculates Reactive power (Q).

    \\[Q = \\text{rms}(V) \\times \\text{rms}(I) \\times \\sin(\\phi)\\]

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.
        phase_shift: (n_samples, 1)-dimensional array of phase shifts.

    Returns:
        Reactive power as a (n_samples, 1)-dimensional array.
    """
    return rms(voltage) * rms(current) * np.sin(phase_shift)


def apparent_power(voltage, current):
    """Calculates Apparent power (S).

    \\[S = \\text{rms}(V) \\times \\text{rms}(I)\\]

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.
        phase_shift: (n_samples, 1)-dimensional array of phase shifts.

    Returns:
        Apparent power as a (n_samples, 1)-dimensional array.
    """
    return rms(voltage) * rms(current)


def vi_trajectory(voltage, current, num_samples=20, window_length_periods=10,
                  normalize=True):
    """Calculates the VI-trajectory.

    Averages the first window_length_periods periods of measurements, extracts
    n_samples equidistant sampling points, and normalizes by dividing by
    maximum.

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.
        num_samples (int): Number of sampling points used.
        window_length_periods (int): Number of periods used for sampling
        normalize (bool): Whether to normalize (divide by max).

    Returns:
        VI-trajectory as tuple of (n_samples, 20)-dimensional arrays of normalized voltage and current values.
    """
    # Calculate equidistant sampling points within each period
    sample = np.linspace(0, PERIOD_LENGTH-1, num=num_samples,
                         endpoint=False, dtype='int')

    # Calculate V mean of each sampling point over window_length_periods periods
    voltage_windows = [voltage[:, i*PERIOD_LENGTH + sample]
                       for i in range(window_length_periods)]
    v = np.dstack(voltage_windows)
    v_mean = np.mean(v, axis=2)

    # Calculate I mean of each sampling point over window_length_periods periods
    current_windows = [current[:, i*PERIOD_LENGTH + sample]
                       for i in range(window_length_periods)]
    i = np.dstack(current_windows)
    i_mean = np.mean(i, axis=2)

    # Normalize voltage and current to [0,1] by dividing by range
    if normalize:
        v_mean = np.divide(v_mean, np.max(
            np.abs(v_mean), axis=1).reshape(-1, 1))
        i_mean = np.divide(i_mean, np.max(
            np.abs(i_mean), axis=1).reshape(-1, 1))
    return v_mean, i_mean


def form_factor(current):
    """Calculates Form factor (FF).

    \\[FF = \\frac{\\text{rms}(I)}{\\text{mean}(|I|)}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Form factor as a (n_samples, 1)-dimensional array.
    """
    return rms(current) / np.mean(np.abs(current), axis=1).reshape(-1, 1)


def crest_factor(current):
    """Calculates Crest factor (CF).

    \\[CF = \\frac{\\max(|I|)}{\\text{rms}(I)}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Crest factor as a (n_samples, 1)-dimensional array.
    """
    return np.max(np.abs(current), axis=1).reshape(-1, 1) / rms(current)


def resistance_mean(voltage, current):
    """Calculates Resistance \\(R_\\text{mean}\\) (mean version) from
    voltage and current arrays.

    \\[R_\\text{mean} = \\frac{\\sqrt{\\text{mean}(V^2)}}{\\sqrt{\\text{mean}(I^2)}}\\]

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Resistance as a (n_samples, 1)-dimensional array.
    """
    numerator = np.sqrt(np.mean(np.square(voltage), axis=1))
    denominator = np.sqrt(np.mean(np.square(current), axis=1))
    return (numerator / denominator).reshape(-1, 1)


def resistance_median(voltage, current):
    """Calculates Resistance \\(R_\\text{median}\\) (median version) from
    voltage and current arrays.

    \\[R_\\text{median} = \\frac{\\sqrt{\\text{median}(V^2)}}{\\sqrt{\\text{median}(I^2)}}\\]

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Resistance as a (n_samples, 1)-dimensional array.
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
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Admittance as a (n_samples, 1)-dimensional array.
    """
    return 1 / resistance_mean(voltage, current)


def admittance_median(voltage, current):
    """Calculates Admittance \\(Y_\\text{median}\\) (median version) from
    voltage and current arrays.

    \\[Y_\\text{median} = \\frac{1}{R_\\text{median}} =
    \\frac{\\sqrt{\\text{median}(I^2)}}{\\sqrt{\\text{median}(V^2)}}\\]

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Admittance as a (n_samples, 1)-dimensional array.
    """
    return 1 / resistance_median(voltage, current)


def log_attack_time(current):
    """Calculates the LogAttackTime \\(\\ln(\\underset{t}{\\arg \\max}(I_t))\\).

    Unit used: ms.

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        LogAttackTime as a (n_samples, 1)-dimensional array.
    """
    starting_times = np.argmax(current, axis=1).reshape(-1, 1)

    # Make sure input to log is > 0
    return np.log(np.where(starting_times > 0, starting_times, 1) / 6.4)


def temporal_centroid(current):
    """Calculates the Temporal centroid \\(C_t\\).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements. Then \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\) and
    \\[C_t = \\frac{1}{f_0} \\cdot \\frac{\\sum_{k=1}^N I_{P(k)} \\cdot k}
       {\\sum_{k=1}^N I_{P(k)}} \\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Temporal centroid as a (n_samples, 1)-dimensional array.
    """
    n_samples = current.shape[0]

    # Reshape into (n_samples, n_periods, length_period)
    iw = current.reshape(n_samples, -1, PERIOD_LENGTH)

    # Calculate RMS for each period
    ip = np.sqrt(np.mean(np.square(iw), axis=2))

    # Calculate numerator and denominator and put together
    numerator = np.sum(ip * np.arange(1, ip.shape[1]+1), axis=1)
    denominator = np.sum(ip, axis=1)
    return 50 * (numerator / denominator).reshape(-1, 1)


def inrush_current_ratio(current):
    """"Calculates the Inrush current ratio (ICR).

    Let \\(I_{W(k)}\\) denote the \\(k\\)th of \\(N\\) periods of current
    measurements. Then \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\) and

    \\[ICR =  \\frac{I_{P(1)}}{I_{P(N)}}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Inrush current ratio as a (n_samples, 1)-dimensional array.
    """
    return rms(current[:, :128]) / rms(current[:, -128:])


def _positive_negative_half_cycle_ratio_single(current):
    """Calculates Positive-negative half cycle ratio (PNR).

    Let \\(I_{P_\text{pos}}\\) and \\(I_{P_\text{neg}}\\) be the RMS of 10
    averaged positive and negative current half cycles. Then

    \\[PNR = \\frac{\\min\\{I_{P_\\text{pos}}, I_{P_\\text{neg}}\\}}
                   {\\max\\{I_{P_\\text{pos}}, I_{P_\\text{neg}}\\}}\\]

    Args:
        current: (window_size, )-dimensional array of current measurements.

    Returns:
        PNR as a float.
    """
    # Create indices for positive and negative half cycles of sin wave
    idcs = np.arange(0, PERIOD_LENGTH * 10)
    p_idcs = idcs[idcs % PERIOD_LENGTH < PERIOD_LENGTH / 2]
    n_idcs = idcs[idcs % PERIOD_LENGTH >= PERIOD_LENGTH / 2]

    p_current = current[p_idcs].reshape(10, int(PERIOD_LENGTH / 2))
    n_current = current[n_idcs].reshape(10, int(PERIOD_LENGTH / 2)) * -1

    # Calculate rms of averaged half cycles
    p_current_rms = rms(np.mean(p_current, axis=0), axis=0)
    n_current_rms = rms(np.mean(n_current, axis=0), axis=0)

    return min(p_current_rms, n_current_rms) / max(p_current_rms, n_current_rms)


def positive_negative_half_cycle_ratio(current):
    """Calculates Positive-negative half cycle ratio (PNR).

    Let \\(I_{P_\\text{pos}}\\) and \\(I_{P_\\text{neg}}\\) be the RMS of 10
    averaged positive and negative current half cycles. Then

    \\[PNR = \\frac{\\min\\{I_{P_\\text{pos}}, I_{P_\\text{neg}}\\}}
                   {\\max\\{I_{P_\\text{pos}}, I_{P_\\text{neg}}\\}}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        PNR as a (n_samples, 1)-dimensional array.
    """
    return np.apply_along_axis(_positive_negative_half_cycle_ratio_single,
                               1, current).reshape(-1, 1)


def max_min_ratio(current):
    """Calculates the Max-min ratio (MAMI).

    \\[MAMI = \\frac{\\min\\{|\\max(I_\\text{ROI})|, |\\min(I_\\text{ROI})|\\}}
                    {\\max\\{|\\max(I_\\text{ROI})|, |\\min(I_\\text{ROI})|\\}}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Max-min ratio as a (n_samples, 1)-dimensional array.
    """
    extrema = np.vstack((np.abs(np.max(current, axis=1)),
                         np.abs(np.min(current, axis=1))))
    return (np.min(extrema, axis=0) / np.max(extrema, axis=0)).reshape(-1, 1)


def peak_mean_ratio(current):
    """Calculates the Peak-mean ratio (PMR).

    \\[MAMI = \\frac{\\max(|I_\\text{ROI}|}{\\text{mean}(|I_\\text{ROI}|}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Peak-mean ratio as a (n_samples, 1)-dimensional array.
    """
    max = np.max(np.abs(current), axis=1)
    mean = np.mean(np.abs(current), axis=1)
    return (max / mean).reshape(-1, 1)


def max_inrush_ratio(current):
    """Calculates the Max inrush ratio (MIR).

    Let \\(I_{W(k)}\\) be the current measurements of the \\(k\\)th period and
    \\(I_{P(k)} = \\text{rms}(I_{W(k)})\\). Then

    \\[MIR = \\frac{I_{P(1)}}{\\max(|I_{W(1)}|)}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Max inrush ratio as a (n_samples, 1)-dimensional array.
    """
    first_period_rms = rms(current[:, :PERIOD_LENGTH])
    first_period_max = np.max(np.abs(current[:, :PERIOD_LENGTH]), axis=1)
    return first_period_rms / first_period_max.reshape(-1, 1)


def mean_variance_ratio(current):
    """Calculates the Mean variance ratio (MVR).

    \\[MVR = \\frac{\\text{mean}(|I_\\text{ROI}|)}
                   {\\text{var}(|I_\\text{ROI}|)}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Mean variance ratio as a (n_samples, 1)-dimensional array.
    """
    return (np.mean(np.abs(current), axis=1)
            / np.var(np.abs(current), axis=1)).reshape(-1, 1)


def waveform_distortion(current):
    """Calculates the Waveform distortion WFD.

    Let \\(I_\\text{avg}^{10}\\) be the current measurements (aligned with the
    rising zero crossing) averaged over the first ten periods and normalized
    (with their maximum value). Let \\(Y_{\\sin} = \\left(\\sin(\\frac{i}{128}
    \\times 2 \\pi)\\right)_{i \\in (0, 1, ..., 127)}\\). Then:

    \\[WFD = \\sum \\left( |Y_{\\sin}| - |I_\\text{avg}^{10}| \\right)\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Waveform distortion as a (n_samples, 1)-dimensional array.
    """
    # TODO: Check in with Daniel about normalizing with max instead of RMS
    current = normalize(average_periods(current, 10), method='max')
    y = np.sin(np.linspace(0, 2*np.pi, 128))
    return np.sum(np.abs(current) - np.abs(y), axis=1).reshape(-1, 1)


def waveform_approximation(current):
    """Calculates the Waveform approximation (WFA).

    Let \\(I_\\text{avg}^{10}\\) be the current measurements (aligned with the
    rising zero crossing) averaged over the first ten periods and normalized
    (with their maximum value). Let \\(S = (0, 6, 12, 19, ..., 114, 120)\\) be
    a list of 20 equidistant sampling points over one period. Then:

    \\[WFA = \\left( I_{\\text{avg}, s}^{10} \\right)_{s \\in S}\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Waveform approximation as a (n_samples, 20)-dimensional array.
    """
    # TODO: Check in with Daniel about normalizing with max instead of RMS
    current = normalize(average_periods(current, n_periods=10), method='max')
    sampling_points = np.linspace(0, 127, num=20, endpoint=False, dtype='int')
    return current[:, sampling_points]


def current_over_time(current):
    """Calculates the Current over time (COT).

    Let \\(I_\\text{rms}^{(k)}\\) be the RMS of the current measurements of
    the \\(k\\)th period. Then:

    \\[COT = \\left( I_\\text{rms}^{(1)}, I_\\text{rms}^{(2)}, ...,
    I_\\text{rms}^{(25)} \\right)\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Current over time as a (n_samples, 25)-dimensional array.
    """
    return helpers.apply_to_periods(current, rms, 25, (0))


def admittance_over_time(voltage, current):
    """Calculates the Admittance over time (COT).

    Let \\(I_\\text{rms}^{(k)}\\) and \\(V_\\text{rms}^{(k)}\\) be the RMS of
    the current and voltage measurements of the \\(k\\)th period. Then:

    \\[AOT = \\left( \\frac{I_\\text{rms}^{(1)}}{V_\\text{rms}^{(1)}},
    \\frac{I_\\text{rms}^{(2)}}{V_\\text{rms}^{(2)}}, ...,
    \\frac{I_\\text{rms}^{(25)}}{V_\\text{rms}^{(25)}} \\right)\\]

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Admittance over time as a (n_samples, 25)-dimensional array.
    """
    return (helpers.apply_to_periods(current, rms, 25, (0)) /
            helpers.apply_to_periods(voltage, rms, 25, (0)))


def periods_to_steady_state_current(current):
    """Calculates the Periods to steady state current (PSS).

    Let \\(I_\\text{rms}^{(k)}\\) be the RMS of the current measurements of
    the \\(k\\)th period. Then:

    \\[L = \\frac{1}{8} \\cdot (\\max(COT) - \\text{median}(COT)) +
    \\text{median}(COT)\\]
    \\[PSS = \\underset{k}{\\arg \\min} \\thinspace I_\\text{rms}^{(k)} > L\\]

    Args:
        current: (n_samples, window_size)-dimensional array of current measurements.

    Returns:
        Periods to steady state current as a (n_samples, 1)-dimensional array.
    """
    cot = current_over_time(current)
    l = (1/8 * (np.max(cot, axis=1) - np.median(cot, axis=1)) +
         np.median(cot, axis=1))
    return np.argmax(cot > l.reshape(-1, 1), axis=1).reshape(-1, 1) + 1