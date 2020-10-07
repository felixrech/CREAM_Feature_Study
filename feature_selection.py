import copy
import scipy
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, plot_confusion_matrix

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier


line_del = ''.join(itertools.repeat('\r', 50))


def normalize_features(features):
    """Use StandardScaler to standardize each feature in feature dictionary.
    """
    scaler = StandardScaler()
    for feature in features:
        features[feature] = scaler.fit_transform(features[feature])
    return features


def print_feature_info(features):
    """Print a table with feature name and dimension for features in features
    dictionary.
    """
    print(f"Total of {len(features)} features:\n")
    dims = [features[feature].shape[1] for feature in features]
    features = pd.DataFrame({'name': list(features.keys()), 'dimension': dims})
    print(features.sort_values('dimension').to_string(index=False))


def _evaluate_feature(X, y, metric, classifier, param_grid):
    """Evaluates a feature via GridSearchCV on param_grid, then calculating
    the mean five-fold cross validation score with given metric.
    """
    cv = GridSearchCV(classifier, param_grid, n_jobs=-1)
    cv.fit(X, y)
    score = np.mean(cross_val_score(cv, X, y, scoring=metric, n_jobs=-1))
    return score, cv.best_estimator_


def _remove_features_by_dimension(features, dimension):
    """Removes features with dimension greater than given value from feature dictionary.
    """
    remove_list = []
    for feature in features:
        if features[feature].shape[1] > dimension:
            remove_list.append(feature)
    for feature in remove_list:
        features.pop(feature)
    return features


def stepwise_regression(features, y, feature_eval, max_features=None,
                        limit_one_dimensional=False):
    """Uses the stepwise regression algorithm to approximate the best feature
    combination for given classifier.

    Args:
        features (dict): Dictionary of features in form {'f_name': (n_samples, f_dim)-array, ...}.
        y (numpy.ndarray): (n_samples, )-dimensional array of true labels
        feature_eval (function): Evaluation function as returned by get_evaluation(..).
        max_features (int): Only consider feature combinations of at most max_features features.
        limit_one_dimensional (bool): Only consider one-dimensional features.

    Returns:
        pandas.core.series.Series: Best feature combination, has labels 'features', 'score', and 'estimator' (already fitted).
    """
    # Set maximum max_features if not specified
    if max_features is None:
        max_features = len(features)

    # Limit features to one-dimensional features if specified
    if limit_one_dimensional:
        # Copy features to not change the argument
        # TODO: Check that this is enough
        features = _remove_features_by_dimension(copy.deepcopy(features), 1)

    # Only for first iteration
    df = pd.DataFrame({'features': [[]], 'score': [0]})

    while True:
        best = df.iloc[0]
        df = pd.DataFrame(
            columns=['change_type', 'change', 'features', 'score', 'estimator']
        )

        # Add possible feature additions to dataframe
        if len(best.features) < max_features:
            for new_feature in set(features) - set(best.features):
                new_features = best.features + [new_feature]
                df.loc[len(df)] = ('addition of', new_feature,
                                   new_features, 0., None)

        # Add possible feature removals to dataframe
        if len(best.features) > 1:
            for rem_feature in best.features:
                new_features = list(set(best.features) - set([rem_feature]))
                df.loc[len(df)] = ('removal of', rem_feature,
                                   new_features, 0., None)

        # Calculate score for each entry of dataframe
        for i in range(len(df)):
            print(f"{line_del}Considering {df.loc[i, 'change_type']}"
                  f" {df.loc[i, 'change']}...", end='')
            feature_comb = np.hstack([features[f] for f in df.loc[i].features])
            df.loc[i, 'score'], df.loc[i, 'estimator'] = feature_eval(
                feature_comb, y)

        # Print overview
        df = df.sort_values('score', ascending=False)
        print(f"{line_del}{df[['change_type', 'change', 'score']]}")

        # Handle feature removal/addition or end of iteration
        new_best = df.iloc[0]
        if new_best.score > best.score:
            print(f"{len(new_best.features)} Feature(s) selected with score "
                  f"{round(new_best.score, 3)}:\t{new_best.features}\n")
        else:
            print("No improvement, returning")
            return df.iloc[0]


def get_evaluation(model, custom_params={}):
    """Returns a evaluation function partially evaluated with default
    parameters for param_grid and metric, as well as given classifier type.
    """
    if model == 'knn':
        classifier = KNeighborsClassifier(n_jobs=-1)
        param_grid = {'n_neighbors': np.arange(1, 20, 2)}
    elif model == 'svc':
        classifier = SVC()
        param_grid = {'gamma': [0.1, 1, 10, 100]}
    elif model == 'ridge':
        classifier = RidgeClassifier()
        param_grid = {'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
    elif model == 'decision_tree':
        classifier = DecisionTreeClassifier()
        param_grid = {'max_depth': np.arange(1, 20, 2),
                      'min_samples_split': [5, 10, 50]}
    elif model == 'adaboost':
        classifier = AdaBoostClassifier()
        param_grid = {'n_estimators': [2, 5, 10, 20, 50]}
    else:
        raise ValueError(f"Classifier of type '{model}' not implemented")

    return partial(_evaluate_feature, metric='f1_macro', classifier=classifier,
                   param_grid={**param_grid, **custom_params})


def simple_stepwise_regression(n_features, y, model, max=None, one_dim=False,
                               custom_params={}):
    """Ease-of use interface to the stepwise regression function.

    Args:
        n_features (dict): Dictionary of normalized features in form {'name': np.ndarray, ...}.
        y (numpy.ndarray): Array containing the true labels.
        model (str): String with the classifier to use (knn, svc, ridge, decision_tree, or adaboost).
        max (int): Maximum number of features to use.
        one_dim (bool): Whether to only use one-dimensional features.
        custom_params (dict): Dictionary of additional parameters for grid search.

    Returns:
        pandas.core.series.Series: Best feature combination, has labels 'features', 'score', and 'estimator' (already fitted).
    """
    evaluation = get_evaluation(model, custom_params)
    return stepwise_regression(n_features, y, evaluation, max, one_dim)


def feature_combinations(normed_features, y,  n_features, metric='f1_macro',
                         models=['knn', 'svc', 'ridge', 'decision_tree']):

    df = pd.DataFrame(columns=['model', 'feature_combination', 'dimension',
                               'score'])
    # Generator empty after first model if not converted to list
    feature_combs = list(itertools.combinations(normed_features.keys(),
                                                n_features))
    num_combs = int(scipy.special.comb(len(normed_features), n_features))
    start = datetime.now()

    for j, model in enumerate(models):
        evaluation = get_evaluation(model)

        for i, combination in enumerate(feature_combs):
            now_num = j*num_combs + i + 1
            remaining = (len(models)-(j+1)) * num_combs + num_combs - i
            time_remaining = (datetime.now() - start) / now_num * remaining
            print(f"{line_del}Model {j+1}/{len(models)}, "
                  f"feature combination {i+1}/{num_combs}, "
                  f"time remaining: {str(time_remaining)[:-7]}", end='')

            X = np.empty((y.shape[0], 0))
            for feature in combination:
                X = np.hstack((X, normed_features[feature]))
                score, _ = evaluation(X, y)
                df.loc[now_num] = (model, combination, X.shape[1], score)

    total_time = (datetime.now() - start)
    print(f'{line_del}Calculations finished in {total_time}', end='')

    return df.sort_values('score', ascending=False)
