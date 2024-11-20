import argparse

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import utils


def train_RandomForestClassifier(X_train, y_train, params):
    """
    Train a RandomForestClassifier with the given parameters.
    :param X_train: Feature data for training
    :param params: Dictionary containing the params for RandomForestClassifier
    :return: Trained RandomForestClassifier model
    """
    # Extract PCA components if specified
    n_components = params.get("PCA", None)
    if n_components is not None:
        print(f"\tapplying PCA with {n_components} components")
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)

    # Extract RandomForest parameters
    rf_params = params.get("CLS", {})
    rf_params.setdefault("random_state", 42)  # Ensure reproducibility
    rf_params.setdefault("class_weight", "balanced")

    # Train the RandomForestClassifier
    print("\ttraining cls with parameters: \n", rf_params)
    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)
    return clf


def train_StackingClassifier(X_train, y_train, params):
    """
    Train a StackingClassifier with the given parameters.
    :param X_train: Feature data for training
    :param params: Dictionary containing the parameters for StackingClassifier
    :return: Trained StackingClassifier model
    """
    # Extract PCA components if specified
    n_components = params.get("PCA", None)
    cls_params = params['CLS']
    random_state = params['CLS'].get('random_state', 42)
    class_weight = params['CLS'].get('class_weight', 'balanced')

    rf_params = {
        key.replace('rf__', ''): value
        for key, value in cls_params.items() if key.startswith('rf__')
    }
    gb_params = {
        key.replace('gb__', ''): value
        for key, value in cls_params.items() if key.startswith('gb__')
    }
    svm_params = {
        key.replace('svm__', ''): value
        for key, value in cls_params.items() if key.startswith('svm__')
    }
    meta_params = {
        key.replace('final_estimator__', ''): value
        for key, value in cls_params.items()
        if key.startswith('final_estimator__')
    }

    rf_params.setdefault('random_state', random_state)
    rf_params.setdefault('class_weight', class_weight)
    gb_params.setdefault('random_state', random_state)
    svm_params.setdefault('random_state', random_state)
    meta_params.setdefault('random_state', random_state)
    meta_params.setdefault('class_weight', class_weight)

    if n_components is not None:
        print(f"\tapplying PCA with {n_components} components")
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Define base estimators
    base_estimators = [
        ('rf', RandomForestClassifier(**rf_params)),
        ('gb', GradientBoostingClassifier(**gb_params)),
        ('svm', SVC(**svm_params))
    ]

    # Define meta-estimator
    meta_estimator = LogisticRegression(**meta_params)

    # Train the StackingClassifier
    print("\ttraining cls with parameters:")
    print("\t - rf:", rf_params)
    print("\t - gb:", gb_params)
    print("\t - svm:", svm_params)
    print("\t - lg:", meta_params)
    clf = StackingClassifier(estimators=base_estimators,
                             final_estimator=meta_estimator,
                             passthrough=True)
    clf.fit(X_train, y_train)
    return clf


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and save it")
    parser.add_argument('-i', '--params-path', type=str, required=True,
                        help="Path to the JSON file with params")
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help="Output file to save the model.pkl")
    parser.add_argument('-t', '--train-csv', type=str, nargs='+',
                        help="Path(s) to the train CSV file(s)")
    parser.add_argument('-c', '--classifier', choices=['rforest', 'stacking'],
                        required=True,
                        help="Train either 'rforest' or 'stacking' classifier")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    print("inference: loading params")
    params = utils.json_load(args.params_path)

    print("inference: loading data")
    X_train, y_train = None, None
    for path in args.train_csv:
        try:
            x, y = utils.load_data(path)
            if X_train is None:
                X_train, y_train = x, y
            else:
                X_train = np.vstack((X_train, x))
                y_train = np.hstack((y_train, y))
        except Exception as e:
            print(f"Error loading data from {path}: {e}")
            exit(1)

    if args.classifier == "rforest":
        print("inference: training 'RandomForestClassifier'")
        cls = train_RandomForestClassifier(X_train, y_train, params)
    elif args.classifier == "stacking":
        print("inference: training 'StackingClassifier'")
        cls = train_StackingClassifier(X_train, y_train, params)
    else:
        print("inference: Unknown classifier")
        exit(1)

    print("inference: saving model")
    utils.mkdir(utils.path_dirname(args.output_file))
    utils.save_model(cls, args.output_file)

    print("inference: done")

