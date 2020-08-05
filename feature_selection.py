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
    scaler = StandardScaler()
    for feature in features:
        features[feature] = scaler.fit_transform(features[feature])
        # if features[feature].shape[1] == 1:
        #     features[feature] = scaler.fit_transform(features[feature])
        # else:
        #     features[feature] -= np.mean(features[feature], axis=(0, 1))
        #     features[feature] /= np.std(features[feature], axis=(0,1))
    return features


def print_feature_info(features):
    print(f"Total of {len(features)} features:\n")
    dims = [features[feature].shape[1] for feature in features]
    features = pd.DataFrame({'name': list(features.keys()), 'dimension': dims})
    print(features.sort_values('dimension').to_string(index=False))


def evaluate_feature(X, y, metric, classifier, param_grid):
    cv = GridSearchCV(classifier, param_grid, n_jobs=-1)
    cv.fit(X, y)
    score = np.mean(cross_val_score(cv, X, y, scoring=metric, n_jobs=-1))
    return score, cv.best_estimator_


def remove_features_by_dimension(features, dimension):
    remove_list = []
    for feature in features:
        if features[feature].shape[1] > dimension:
            remove_list.append(feature)
    for feature in remove_list:
        features.pop(feature)
    return features


def forward_selection(features, y, feature_eval, max_features=None,
                      limit_one_dimensional=False):
    # Set maximum max_features if not specified
    if max_features is None:
        max_features = len(features)

    # Limit features to one-dimensional features if specified
    if limit_one_dimensional:
        features = remove_features_by_dimension(features, 1)

    # Set up variables
    df = pd.DataFrame({'features': itertools.repeat([], len(features)),
                       'score': itertools.repeat(0, len(features))})

    # Add a maximum of max_features features
    for n_features in range(max_features):
        old_df = df
        best = df.iloc[0]
        df = pd.DataFrame(columns=['new', 'features', 'score', 'estimator'])

        # Calculate score for all remaining unselected features
        for i, new_feature in enumerate(set(features) - set(best.features)):
            print(f"{line_del}\tLooking at new feature: {new_feature}", end='')
            new_features = best.features + [new_feature]
            comb_features = np.hstack([features[f] for f in new_features])
            score, estimator = feature_eval(comb_features, y)
            df.loc[i] = (new_feature, new_features, score, estimator)

        # Prepare for next iteration
        df = df.sort_values('score', ascending=False)
        new_best = df.iloc[0]
        if new_best.score > best.score:
            print(f"{line_del}\n{df[['new', 'score']]}\n\n"
                  f"{len(new_best.features)} Feature(s) selected with score "
                  f"{round(new_best.score, 3)}:\t{new_best.features}\n")
        else:
            print(f"{line_del}\n{df[['new', 'score']]}\n\n"
                  "No new feature selected, returning")
            break

    return df.iloc[0]


def get_evaluation(model, custom_params):
    if model == 'knn':
        classifier = KNeighborsClassifier(n_jobs=-1)
        param_grid = {'n_neighbors': np.arange(1, 20)}
    elif model == 'svc':
        classifier = SVC()
        param_grid = {'gamma': [0.1, 1, 10, 100]}
    elif model == 'ridge':
        classifier = RidgeClassifier()
        param_grid = {'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
    elif model == 'decision_tree':
        classifier = DecisionTreeClassifier()
        param_grid = {'max_depth': range(1, 20, 2),
                      'min_samples_split': [5, 10, 50, 100]}
    elif model == 'adaboost':
        classifier = AdaBoostClassifier()
        param_grid = {'n_estimators': [2, 5, 10, 20, 50]}
    else:
        raise ValueError(f"Classifier of type {model} not implemented")

    return partial(evaluate_feature, metric='f1_macro', classifier=classifier,
                   param_grid={**param_grid, **custom_params})


def simple_forward_selection(n_features, y, model, max=None, one_dim=False,
                             custom_params={}):
    evaluation = get_evaluation(model, custom_params)
    return forward_selection(n_features, y, evaluation, max, one_dim)


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
