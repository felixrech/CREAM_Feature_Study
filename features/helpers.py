"""Implements some helper functions for features.
"""


import numpy as np
from scipy.stats import gmean
import matplotlib.pyplot as plt


import feature_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, plot_confusion_matrix


PERIOD_LENGTH = 128


##############################################################################
#                                                                            #
#                START FEATURE EVALUATION HELPER FUNCTIONS                   #
#                                                                            #
##############################################################################

def feature_boxplot(title, X, y):
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
                   labels=['Heater', 'Millingplant', 'Pump'])
    fig.suptitle(title)
    plt.show()


def feature_evaluation(X, y):
    # Standardize feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Prepare a kNN
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(1, 20)}

    # Do cross-validation on kNN
    score, estimator = feature_selection.evaluate_feature(X, y, 'f1_macro',
                                                          knn, param_grid)

    # Predict the data set
    y_pred = estimator.predict(X)

    # Print accuracy, precision, recall, and f1-score
    print("Cross-validation F1-score: {:.2f}".format(score))
    # print("\nPerformance on training data:\n")
    # print(classification_report(y, y_pred))

    # Plot confusion matrix (absolute & relative)
    # plot_confusion_matrix(estimator, X, y)
    # plt.show()
    # plot_confusion_matrix(estimator, X, y, normalize='true')
    # plt.show()


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


def average_periods(X, n_periods):
    X = X[:, : n_periods * PERIOD_LENGTH]
    X = X.reshape(X.shape[0], n_periods, PERIOD_LENGTH)
    return np.mean(X, axis=1)


def normalize(X, method='max'):
    if method == 'max':
        return X / np.max(X, axis=1).reshape(-1, 1)
    raise ValueError("Chosen method type does not exist, "
                     "please refer to the docstring for available methods!")


def apply_to_periods(X, func, n_periods, args):
    X = X[:, : n_periods * PERIOD_LENGTH]
    X = X.reshape(X.shape[0], n_periods, PERIOD_LENGTH)
    return np.apply_along_axis(func, 2, X, args)
