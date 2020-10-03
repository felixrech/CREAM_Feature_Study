import csv
import sys
import random
import numpy as np


PATH_TO_DATA = '/var/lib/cream/'
PATH_TO_CREAM = PATH_TO_DATA + 'CREAM/'

POWER_FREQUENCY = 50  # Hz
SAMPLING_RATE = 6400  # Measurements / second
PERIOD_LENGTH = SAMPLING_RATE // POWER_FREQUENCY


##############################################################################
#                                                                            #
#                         START DATA ACCESS FUNCTIONS                        #
#                                                                            #
##############################################################################


def get_component_events(filter=True):
    # Import data utility from the CREAM repo
    sys.path.append(PATH_TO_CREAM)
    from data_utility import CREAM_Day

    cream = CREAM_Day(PATH_TO_DATA + '2018-08-23/')

    events = cream.load_component_events(PATH_TO_DATA + "component_events.csv",
                                         filter_day=False)
    if not filter:
        return events
    else:
        return events[((events.Component != 'unlabeled') &
                       (events.Event_Type == 'On'))]


def read_event(id, duration):
    with open(f"{PATH_TO_DATA}component_events/{id}.csv", 'r') as f:
        reader = csv.reader(f)
        voltage = np.array(next(reader)[:int(duration*6400)]).astype(float)
        current = np.array(next(reader)[:int(duration*6400)]).astype(float)
    return voltage, current


def read_dataset(component_events, start_offset=0, duration=2):
    event_types, voltages, currents = [], [], []

    idcs = ((component_events.Component != 'unlabeled') &
            (component_events.Event_Type == 'On'))
    full_duration = duration + (start_offset / 6400)

    for event in component_events[idcs].itertuples():
        # Print crude progress report
        progress_percent = round(100 * ((len(event_types)+1) / 1499), 1)
        print(f"\r\r\r\r\r\r{progress_percent}%",  end='', flush=True)
        # Load current from csv file
        voltage, current = read_event(event.ID, duration=full_duration)
        # Append voltage, current and event to lists
        event_types.append(event.Component)
        voltages.append(voltage[start_offset:])
        currents.append(current[start_offset:])

    # Transform harmonics and corresponding events into correct format
    voltage = np.vstack(voltages)
    current = np.vstack(currents)
    y = np.array(event_types).reshape(-1,)

    return voltage, current, y


def get_events_sample(size, events=None, seed=42, duration=4):
    if events is None:
        events = get_component_events()
        events = events[(events.Component != 'unlabeled') &
                        (events.Event_Type == 'On')]
    random.seed(42)
    samples_idcs = random.sample(range(len(events)), size)

    events = [events.iloc[idx] for idx in samples_idcs]
    return [(event, read_event(event.ID, duration)) for event in events]


##############################################################################
#                                                                            #
#                     START DATA PREPROCESSING FUNCTIONS                     #
#                                                                            #
##############################################################################


def preprocess(voltage, current, periods, offsets=False):
    """Preprocesses data by removing area without activity and aligning with rising zero crossing

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.
        periods (int): Length of area of interest in periods.
        offsets (bool): Whether to return offsets used (True) or only measurements.

    Returns:
        Tuple of preprocessed measurements as (n_samples, periods*128)-dimensional array (& offsets if specified).
    """
    voltage, current, event_offsets \
        = offset_correct(voltage, current, periods+1)
    voltage, current, period_offsets \
        = synchronize_period(voltage, current, periods=periods)
    if not offsets:
        return voltage, current
    return voltage, current, event_offsets, period_offsets


def _offset_correct_single(voltage, current, periods, pre=5):
    """Calculates the offset at which nonactivity ends (for 1D V,I arrays).

    Nonactivity has been observed to have absolute current values smaller than
    1 ampere, so this is used to detect first occurrence of larger values.

    Args:
        voltage: (window_size,)-dimensional array of voltage measurements.
        current: (window_size,)-dimensional array of current measurements.
        periods (int): Length of area of interest in periods.
        pre (int): is subtracted from actual detected offset (for period offsetting).

    Returns:
        Tuple containing AOI of voltage and current and the offset used.
    """
    event_offset = max(np.min(np.where(np.abs(current) > 1)[0]) - pre, 0)
    if event_offset+periods*PERIOD_LENGTH > len(current):
        raise ValueError("Use longer input for offsetting (event_offset + "
                         f"periods*128 = {event_offset + periods*128} > "
                         f"{len(current)})")
    return (voltage[event_offset: event_offset+periods*PERIOD_LENGTH],
            current[event_offset: event_offset+periods*PERIOD_LENGTH],
            event_offset)


def offset_correct(voltage, current, periods=51, pre=5):
    """Calculates the offset at which nonactivity ends.

    Nonactivity has been observed to have absolute current values smaller than
    1 ampere, so this is used to detect first occurrence of larger values.

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.
        periods (int): Length of area of interest in periods. Default is 51.
        pre (int): is subtracted from actual detected offset (for period offsetting).

    Returns:
        Tuple containing arrays with AOI of voltage and current and the offset used.
    """
    v = np.empty((voltage.shape[0], periods * PERIOD_LENGTH))
    i = np.empty((voltage.shape[0], periods * PERIOD_LENGTH))
    offsets = []

    for idx in range(voltage.shape[0]):
        v[idx], i[idx], offset = _offset_correct_single(voltage[idx],
                                                        current[idx],
                                                        periods, pre)
        offsets.append(offset)
    return v, i, np.array(offsets).reshape(-1, 1)


def _synchronize_period_single(voltage, current, periods, use_periods=10):
    """Calculates offset that aligns current with rising zero crossing (for 1D V,I arrays).

    Offset is calculated as the mean of the first use_periods zero crossings.

    Args:
        voltage: (window_size,)-dimensional array of voltage measurements.
        current: (window_size,)-dimensional array of current measurements.
        periods (int): Length of area of interest in periods.
        use_periods (int): Number of zero crossings to average for offset calculation.

    Returns:
        Tuple containing arrays with AOI of voltage and current and the offset used.
    """
    # Extract area of interest
    x = current[: (periods+1)*PERIOD_LENGTH]

    # Get all zero crossings within the first use_periods periods
    roots = np.mod(np.where(
        (x[:use_periods*PERIOD_LENGTH] <= 0)
        & (x[1:use_periods*PERIOD_LENGTH+1] > 0)
    )[0], 128)  # mod makes sure indices (and their mean) are in [0, 128)
    # Calculate mean of zero crossings and fail to 0 if there are none
    offset = int(np.mean(roots)) + 1 if len(roots) > 0 else 0

    # Return cut area
    return (voltage[offset: offset + periods*PERIOD_LENGTH],
            current[offset: offset + periods*PERIOD_LENGTH],
            offset)


def synchronize_period(voltage, current, periods=-1):
    """Calculates offset that aligns current with rising zero crossing.

    Offset is calculated as the mean of the first use_periods zero crossings.

    Args:
        voltage: (n_samples, window_size)-dimensional array of voltage measurements.
        current: (n_samples, window_size)-dimensional array of current measurements.
        periods (int): Length of area of interest in periods. Default is max possible.
        use_periods (int): Number of zero crossings to average for offset calculation.

    Returns:
        Tuple containing arrays with AOI of voltage and current and the offset used.
    """
    if periods == -1:
        periods = int(np.floor((current.shape[1]-1) / PERIOD_LENGTH))

    v = np.empty((voltage.shape[0], periods * PERIOD_LENGTH))
    i = np.empty((voltage.shape[0], periods * PERIOD_LENGTH))
    offsets = []

    for idx in range(voltage.shape[0]):
        v[idx], i[idx], offset = _synchronize_period_single(voltage[idx],
                                                            current[idx],
                                                            periods=periods)
        offsets.append(offset)
    return v, i, np.array(offsets).reshape(-1, 1)
