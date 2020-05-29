import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, plot_confusion_matrix


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


def forward_selection(features, y, feature_eval, max_features=None,
                      limit_one_dimensional=False):
    # Set maximum max_features if not specified
    if max_features is None:
        max_features = len(features)
    # Limit features to one-dimensional features if specified
    if limit_one_dimensional:
        for feature in features:
            remove_list = []
            if features[feature].shape[1] > 1:
                remove_list.append(feature)
        for feature in remove_list:
            features.pop(feature)
    # Set up variables
    used_features = []
    used_features_arr = np.empty((y.shape[0], 0))
    current_score = 0
    # Add a maximum of max_features features
    for n_features in range(max_features):
        # Announce next round in forward selection
        print(f"{n_features} Feature(s) selected with score "
              f"{round(current_score, 3)}:\t{used_features}\n")
        # Calculate score for all remaining unselected features
        scores, estimators = [], []
        for feature in features:
            comb_features = np.hstack((used_features_arr, features[feature]))
            score, estimator = feature_eval(comb_features, y)
            scores.append(score)
            estimators.append(estimator)
        # Build dataframe from scores, then sort and print it
        df = pd.DataFrame({'feature': list(features.keys()), 'score': scores,
                           'estimator': estimators})
        df = df.sort_values('score', ascending=False)
        print(df[['feature', 'score']].to_string(index=False), '\n')
        # Extract the best feature from dataframe
        selected_feature, selected_score, selected_estimator = df.iloc[0]
        # If selected feature would lead to better overall performance
        if selected_score > current_score:
            # Update variables for selection
            current_score, current_estimator = selected_score, selected_estimator
            used_features.append(selected_feature)
            used_features_arr = np.hstack(
                (used_features_arr, features[feature]))
            features.pop(selected_feature)
            print(f"Selecting feature:  {selected_feature}\n\n\n")
        # Otherwise stop iteration
        else:
            print("No new feature selected (no score higher than current score)")
            break

    return current_estimator
