import argparse
import time

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

import utils


def tuning_SVC(X_full, y_full, depth='small', silent=False):
    """
    Train and evaluate an SVC on the provided dataset.
    :param X_full: Training data train+dev
    :param y_full: Training data labels
    :param depth: Tuning depth 'small', 'mid', 'big'
    :param silent: Bool silent output
    :return (best_model, best_params, best_pca_n)
    """
    print("tuning: tuning 'SVC'")

    verbose = 3 if not silent else 1

    param_grids = {
        'small': {
            'C': [0.1, 1],
            'kernel': ['linear', 'rbf'],
        },
        'mid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],
        },
        'big': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
        }
    }
    param_grid = param_grids.get(depth, param_grids['small'])
    pca_components = [50, 150, 250]

    best_score = -np.inf
    best_params = None
    best_model = None
    best_pca_n = None

    for i, n_components in enumerate(pca_components):
        print(f"[loop {i + 1} of {len(pca_components)}]-->")
        print(f"Testing PCA with {n_components} components...")

        start_time = time.time()

        # Apply PCA transformation
        pca = PCA(n_components=n_components)
        X_full_pca = pca.fit_transform(X_full)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=SVC(random_state=42, class_weight='balanced'),
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            verbose=verbose
        )
        grid_search.fit(X_full_pca, y_full)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            best_pca_n = n_components
        elapsed_time = time.time() - start_time
        print(f"--> [time {elapsed_time:.2f} seconds]")

    best_params["random_state"] = 42
    best_params["class_weight"] = "balanced"

    return best_model, best_params, best_pca_n


def tuning_GradientBoostingClassifier(X_full, y_full, depth='small',
                                      silent=False):
    """
    :param X_full: Training data train+dev
    :param y_full: Training data labels
    :param depth: Tuning depth 'small', 'mid', 'big'
    :param silent: Bool silent output
    :return (best_model, best_params, best_pca_n)
    """
    print("tuning: tuning 'GradientBoostingClassifier'")

    verbose = 3 if not silent else 1

    param_grids = {
        'small': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5],
        },
        'mid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.05, 0.01],
            'max_depth': [3, 5, 10],
            'subsample': [0.8, 1.0],
        },
        'big': {
            'n_estimators': [50, 100, 200, 500],
            'learning_rate': [0.1, 0.05, 0.01, 0.005],
            'max_depth': [3, 5, 10, 20],
            'subsample': [0.6, 0.8, 1.0],
            'min_samples_split': [2, 5, 10],
        }
    }
    param_grid = param_grids.get(depth, param_grids['small'])
    pca_components = [50, 150, 250]

    best_score = -np.inf
    best_params = None
    best_model = None
    best_pca_n = None

    for i, n_components in enumerate(pca_components):
        print(f"[loop {i + 1} of {len(pca_components)}]-->")
        print(f"Testing PCA with {n_components} components...")

        start_time = time.time()

        # Apply PCA transformation
        pca = PCA(n_components=n_components)
        X_full_pca = pca.fit_transform(X_full)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_grid=param_grid,
            scoring='f1',
            cv=3,
            verbose=verbose
        )
        grid_search.fit(X_full_pca, y_full)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            best_pca_n = n_components
        elapsed_time = time.time() - start_time
        print(f"--> [time {elapsed_time:.2f} seconds]")

    best_params["random_state"] = 42

    return best_model, best_params, best_pca_n


def tuning_RandomForestClassifier(X_full, y_full, depth='small', silent=False):
    """
    Train and evaluate a RandomForestClassifier on the provided dataset.
    :param X_full: Training data train+dev
    :param y_full: Training data labels
    :param depth: Tuning depth 'small', 'mid', 'big'
    :param silent: Bool silent output
    :return (best_model, best_params, best_pca_n)
    """
    print("tuning: tuning 'RandomForestClassifier'")

    verbose = 3 if not silent else 1

    param_grids = {
        'small': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
        },
        'mid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5],
        },
        'big': {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    }
    param_grid = param_grids.get(depth, param_grids['small'])
    pca_components = [50, 150, 250]

    best_score = -np.inf
    best_params = None
    best_model = None
    best_pca_n = None
    for i, n_components in enumerate(pca_components):
        print(f"[loop {i + 1} of {len(pca_components)}]-->")
        print(f"Testing PCA with {n_components} components...")

        start_time = time.time()

        # Apply PCA transformation
        pca = PCA(n_components=n_components)
        X_full_pca = pca.fit_transform(X_full)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42,
                                             class_weight='balanced'),
            param_grid=param_grid,
            scoring='f1',  # Optimize for F1 score
            cv=3,          # n-fold cross-validation
            verbose=verbose
        )
        grid_search.fit(X_full_pca, y_full)

        # Track the best model
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            best_pca_n = n_components
        elapsed_time = time.time() - start_time
        print(f"--> [time {elapsed_time:.2f} seconds]")

    best_params["random_state"] = 42
    best_params["class_weight"] = "balanced"

    return best_model, best_params, best_pca_n


def tuning_StackingClassifier(X_full, y_full,
                              depth='small', silent=False):
    """
    Train and evaluate a StackingClassifier with multiple base estimators and a
    LogisticRegression meta-estimator.
    :param X_full: Training data train+dev
    :param y_full: Training data labels
    :param depth: Tuning depth 'small', 'mid', 'big'
    :param silent: Bool silent output
    :return (best_model, best_params, best_pca_n)
    """
    # HACK: ignore Warnings
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("tuning: tuning 'StackingClassifier'")

    verbose = 3 if not silent else 1

    param_grids = {
        'small': {
            # 'final_estimator__max_iter': [100, 200],
            'final_estimator__solver': ['lbfgs', 'liblinear'],
            'gb__n_estimators': [50, 100],
            'rf__n_estimators': [50, 100],
            'svm__C': [0.1, 1],
            'svm__kernel': ['linear'],
        },
        'mid': {
            # 'final_estimator__max_iter': [100, 200, 500],
            'final_estimator__solver': ['lbfgs', 'liblinear'],
            'gb__n_estimators': [100, 200],
            'rf__max_depth': [None, 10],
            'rf__n_estimators': [100, 200],
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf'],
        },
        'big': {
            # 'final_estimator__max_iter': [100, 200, 500, 1000],
            'final_estimator__solver': ['lbfgs', 'liblinear'],
            'gb__n_estimators': [100, 200, 500],
            'rf__max_depth': [None, 10, 20],
            'rf__n_estimators': [100, 200, 500],
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf', 'poly'],
        }
    }
    param_grid = param_grids.get(depth, param_grids['small'])
    # pca_components = [50, 150, 250]
    pca_components = [50]  # al menos nos toca esperar unas 2h

    best_score = -np.inf
    best_params = None
    best_model = None
    best_pca_n = None

    for i, n_components in enumerate(pca_components):
        print(f"[loop {i + 1} of {len(pca_components)}]-->")
        print(f"Testing PCA with {n_components} components...")

        start_time = time.time()

        # Apply PCA transformation
        pca = PCA(n_components=n_components)
        X_full_pca = pca.fit_transform(X_full)

        # Stacking Classifier
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42,
                                          class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=100,
                                              random_state=42)),
            ('svm', SVC(probability=True, kernel='linear', random_state=42))
        ]

        meta_estimator = LogisticRegression(max_iter=1000,
                                            solver='lbfgs',
                                            class_weight='balanced',
                                            random_state=42)

        stacking_clf = StackingClassifier(estimators=base_estimators,
                                          final_estimator=meta_estimator,
                                          passthrough=True)

        grid_search = GridSearchCV(
            estimator=stacking_clf,
            param_grid=param_grid,
            scoring='f1',  # Optimize for F1 score
            cv=3,          # n-fold cross-validation
            verbose=verbose
        )
        grid_search.fit(X_full_pca, y_full)

        # Track the best model
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            best_pca_n = n_components

        elapsed_time = time.time() - start_time
        print(f"--> [time {elapsed_time:.2f} seconds]")

    best_params["random_state"] = 42
    best_params["class_weight"] = "balanced"

    return best_model, best_params, best_pca_n


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do tuning for selected cls")
    parser.add_argument('-t', '--train-csv', type=str, required=True,
                        help="Path to the train CSV file")
    parser.add_argument('-d', '--dev-csv', type=str, required=True,
                        help="Path to the dev CSV file")
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help="Output file to save the tuning results (JSON)")
    parser.add_argument('-c', '--classifier', required=True,
                        choices=['rforest', 'stacking', 'gradient', 'svc'],
                        help="Do tuning for {rforest,stacking,gradient,svc}")
    parser.add_argument('-m', '--tuning-depth', choices=['small', 'mid', 'big'],
                        default='small',
                        help="Tuning depth: 'small', 'mid', 'big'; def.: small")
    parser.add_argument('-s', '--silent', action='store_true',
                        help="Silent output")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    print("tuning: loading data")
    X_train, y_train = utils.load_data(args.train_csv)
    X_dev, y_dev = utils.load_data(args.dev_csv)
    # Merge train and dev set to perform tuning
    X_full = np.vstack((X_train, X_dev))
    y_full = np.hstack((y_train, y_dev))

    if args.classifier == "rforest":
        tuning_cls = tuning_RandomForestClassifier
    elif args.classifier == "gradient":
        tuning_cls = tuning_GradientBoostingClassifier
    elif args.classifier == "svc":
        # Standardize features
        # NOTE: good for CSV, LogisticRegression and Stacking, ensures
        #       consistency
        scaler = StandardScaler()
        X_full = scaler.fit_transform(X_full)
        X_dev = scaler.transform(X_dev)
        tuning_cls = tuning_SVC
    elif args.classifier == "stacking":
        scaler = StandardScaler()
        X_full = scaler.fit_transform(X_full)
        X_dev = scaler.transform(X_dev)
        tuning_cls = tuning_StackingClassifier
    else:
        print("tuning: Unknown classifier")
        exit(1)

    model, params, pca_n = tuning_cls(X_full, y_full, depth=args.tuning_depth,
                                      silent=args.silent)

    print("tuning: done")
    # Evaluate on dev set using the best model
    pca = PCA(n_components=pca_n).fit(X_train)
    X_dev_pca = pca.transform(X_dev)
    y_pred = model.predict(X_dev_pca)
    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred)

    print("\ntuning: results")
    print(f"\tBest PCA Components: {pca_n}")
    print(f"\tBest Parameters: {params}")
    print(f"\tDev Set Accuracy: {accuracy:.4f}")
    print(f"\tDev Set F1 Score: {f1:.4f}")

    # Save results to JSON
    results = {
        "PCA": pca_n,
        "CLS": params,
        "score": {
            "accuracy": accuracy,
            "f1": f1
        }
    }

    utils.mkdir(utils.path_dirname(args.output_file))
    utils.json_save(results, args.output_file)

"""
1. Diversity in Base Estimators

    Why: Stacking works best when the base estimators have different strengths
    and weaknesses. If all estimators are too similar, you risk redundancy
    without much gain in predictive performance.
    Recommendations:
        Combine linear models (e.g., LogisticRegression, SGDClassifier) with
        non-linear models (e.g., RandomForestClassifier,
        GradientBoostingClassifier).
        Consider algorithms with different assumptions:
            Tree-based models (e.g., DecisionTreeClassifier, XGBoost)
            Distance-based models (e.g., KNeighborsClassifier)
            Probabilistic models (e.g., Naive Bayes).

2. Choosing the Meta-Estimator

    Why: The meta-estimator combines predictions from base models. It needs to
    generalize well without overfitting to base predictions.
    Recommendations:
        Start with a simple model like LogisticRegression or RidgeClassifier for
        the meta-estimator, as they are robust and less likely to overfit.
        For complex problems, consider GradientBoostingClassifier or
        RandomForestClassifier as meta-estimators, but regularize to avoid
        overfitting.

3. Limit Overfitting

    Why: Stacking is prone to overfitting, especially when the training set is
    small.
    How:
        Use cross-validation predictions to train the meta-estimator rather than
        using raw predictions on the training set.
        Use fewer, well-tuned base models instead of many unoptimized ones.

4. Feature Engineering for Stacking

    Why: The meta-estimator learns from the predictions (or probabilities) of
    the base models. You can also include original features for additional
    context.
    Recommendations:
        Use predict_proba outputs instead of raw predictions from base models,
        if supported.
        Add original features to the meta-estimator input (passthrough=True in
        StackingClassifier).

5. Balancing Computational Cost

    Why: Stacking can be computationally expensive when combining many
    estimators.
    Recommendations:
        Start with lightweight models (LogisticRegression, RandomForest) for
        base estimators to assess feasibility.
        Experiment with heavier models (XGBoost, SVM) later for potentially
        better performance.
"""
