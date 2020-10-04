"""Implements some helper functions for features.
"""


import numpy as np
from scipy.stats import gmean
import matplotlib.pyplot as plt

import feature_selection

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix


PERIOD_LENGTH = 128


##############################################################################
#                                                                            #
#                START FEATURE EVALUATION HELPER FUNCTIONS                   #
#                                                                            #
##############################################################################

def get_all_features(voltage, current):
    # Import here to avoid circular import
    from features import vi, spectral, wavelet

    spec = spectral.spectrum(current)
    harmon = spectral.harmonics(current)
    ps = vi.phase_shift(voltage, current)

    return {
        "Active power": vi.active_power(voltage, current, ps),
        "Reactive power": vi.reactive_power(voltage, current, ps),
        "Apparent power": vi.apparent_power(voltage, current),
        "Phase shift": ps,
        "VI Trajectory": np.hstack(vi.vi_trajectory(voltage, current)),
        "Harmonics (first 20)": harmon,
        "Harmonics energy distribution": spectral.harmonics_energy_distribution(harmon, spec),
        "Spectral flatness": spectral.spectral_flatness(spec),
        "Odd even harmonics ratio": spectral.odd_even_ratio(harmon),
        "Tristimulus": spectral.tristiumulus(harmon),
        "Form factor": vi.form_factor(current),
        "Crest factor": vi.crest_factor(current),
        "Total harmonic distortion": spectral.total_harmonic_distortion(harmon, spec),
        "Resistance (mean)": vi.resistance_mean(voltage, current),
        "Resistance (median)": vi.resistance_median(voltage, current),
        "Admittance (mean)": vi.admittance_mean(voltage, current),
        "Admittance (median)": vi.admittance_median(voltage, current),
        "Log attack time": vi.log_attack_time(current),
        "Temporal centroid": vi.temporal_centroid(current),
        "Spectral centroid": spectral.spectral_centroid(spec, current),
        "Harmonic spectral centroid": spectral.harmonic_spectral_centroid(current),
        "Signal to signal mean ratio": spectral.signal_to_signal_mean_ratio(spec),
        "Inrush current ratio": vi.inrush_current_ratio(current),
        "Positive-negative half cycle ratio": vi.positive_negative_half_cycle_ratio(current),
        "Max-min ratio": vi.max_min_ratio(current),
        "Peak-mean ratio": vi.peak_mean_ratio(current),
        "Max inrush ratio": vi.max_inrush_ratio(current),
        "Mean variance ratio": vi.mean_variance_ratio(current),
        "Waveform distortion": vi.waveform_distortion(current),
        "Waveform approximation": vi.waveform_approximation(current),
        "Current over time": vi.current_over_time(current),
        "Admittance over time": vi.admittance_over_time(voltage, current),
        "Periods to steady state": vi.periods_to_steady_state_current(current),
        "2nd harmonic": spectral.second_harmonic(harmon),
        "Transient steady states ratio": vi.transient_steady_states_ratio(current),
        "Current RMS": vi.current_rms(current),
        "High frequency spectral centroid (zero-type filter)": spectral.high_frequency_spectral_centroid(spec, current, 'zero'),
        "High frequency spectral flatness (zero-type filter)": spectral.high_frequency_spectral_flatness(spec, 'zero'),
        "High frequency spectral mean (zero-type filter)": spectral.high_frequency_spectral_mean(spec, 'zero'),
        "High frequency spectral centroid (linear-type filter)": spectral.high_frequency_spectral_centroid(spec, current, 'linear'),
        "High frequency spectral flatness (linear-type filter)": spectral.high_frequency_spectral_flatness(spec, 'linear'),
        "High frequency spectral mean (linear-type filter)": spectral.high_frequency_spectral_mean(spec, 'linear'),
        "High frequency spectral centroid (quadratic-type filter)": spectral.high_frequency_spectral_centroid(spec, current, 'quadratic'),
        "High frequency spectral flatness (quadratic-type filter)": spectral.high_frequency_spectral_flatness(spec, 'quadratic'),
        "High frequency spectral mean (quadratic-type filter)": spectral.high_frequency_spectral_mean(spec, 'quadratic'),
        "(High frequency) Spectral mean (no high pass filter)": spectral.high_frequency_spectral_mean(spec, 'none'),
        "Wavelet decomposition details coefficients (max level)": wavelet.details_coefficients(current),
        "Wavelet decomposition (1st level) energy": wavelet.first_level_energy(current),
        "Wavelet decomposition (all levels) energy": wavelet.all_decomposition_levels_energy(current),
        "Wavelet decomposition dominant scale": wavelet.dominant_scale(current),
        "Wavelet decomposition energy of time": wavelet.energy_over_time(current)
    }


def feature_boxplot(title, X, y, out=True):
    # Set up grid size
    x_dim = int(np.ceil(X.shape[1] / 5))
    y_dim = X.shape[1] if X.shape[1] <= 5 else 5
    # Create figure
    fig, axs = plt.subplots(x_dim, y_dim, figsize=(y_dim*5, x_dim*4))
    for col in range(X.shape[1]):
        x_idx, y_idx = int(np.floor(col / 5)), col % 5
        # Deal with object, one-dimensional and two-dimensional axs
        ax = axs if x_dim == 1 and y_dim == 1 else (
            axs[col] if x_dim == 1 else axs[x_idx, y_idx])
        # Create boxplot for column col of feature
        ax.boxplot([X[y == 'heater'][:, col], X[y == 'millingplant'][:, col],
                    X[y == 'pump'][:, col]],
                   labels=['Heater', 'Millingplant', 'Pump'], showfliers=out)
    fig.suptitle(title)
    plt.show()


def feature_evaluation(X, y, output=True, confusion=False):
    # Standardize feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Prepare a kNN
    knn = KNeighborsClassifier(n_jobs=-1)
    param_grid = {'n_neighbors': np.arange(1, 20)}

    # Do cross-validation on kNN
    score, estimator = feature_selection.evaluate_feature(X, y, 'f1_macro',
                                                          knn, param_grid)

    # Predict the data set
    y_pred = estimator.predict(X)

    # Print accuracy, precision, recall, and f1-score
    if output:
        print("Cross-validation F1-score: {:.2f}".format(score))
    # print("\nPerformance on training data:\n")
    # print(classification_report(y, y_pred))

    if confusion:
        # Plot confusion matrix (absolute & relative)
        plot_confusion_matrix(estimator, X, y)
        plt.show()
        plot_confusion_matrix(estimator, X, y, normalize='true')
        plt.show()
    return score


##############################################################################
#                                                                            #
#                START FEATURE CALCULATION HELPER FUNCTIONS                  #
#                                                                            #
##############################################################################

def rms(x, axis=1):
    """Calculate root mean square (RMS).
    """
    rms = np.sqrt(np.mean(np.square(x), axis=axis))
    if axis == 1:
        return rms.reshape(-1, 1)
    return rms


def geo_mean(x):
    """Calculate geometric mean.
    """
    return gmean(x, axis=1)


def average_periods(X, n_periods, period_length=PERIOD_LENGTH):
    X = X[:, : n_periods * period_length]
    X = X.reshape(X.shape[0], n_periods, period_length)
    return np.mean(X, axis=1)


def normalize(X, method='max'):
    if method == 'max':
        return X / np.max(np.abs(X), axis=1).reshape(-1, 1)
    raise ValueError("Chosen method type does not exist, "
                     "please refer to the docstring for available methods!")


def apply_to_periods(X, func, n_periods, args, period_length=PERIOD_LENGTH):
    X = X[:, : n_periods * period_length]
    X = X.reshape(X.shape[0], n_periods, period_length)
    return np.apply_along_axis(func, 2, X, args)
