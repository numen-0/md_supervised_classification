import argparse

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import train


def test(cls, X_test, y_test, out_dir, silent=False):
    '''
    eval the model on F1-score and confusion matrix, etc.

    :param cls:     classifier
    :param X_test:  test data (PCA already done)
    :param y_test:  test label
    :param out_dir: output directory where results will be saved.
    :param silent:  suppress log output (def.: False).
    '''
    utils.mkdir(out_dir)
    log_file = utils.path_join(out_dir, "results.txt")
    utils.log_init(log_file)

    def plot_cm(cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=["0", "1"],
                    yticklabels=["0", "1"])

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        cm_plot_path = utils.path_join(out_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path)
        plt.close()

        print(f"Confusion matrix saved to {cm_plot_path}")

    utils.log("rq2: evaluating model", log_file, silent)
    y_pred = cls.predict(X_test)
    fscore = f1_score(y_test, y_pred, average='weighted')
    utils.log(f"    f-score: {fscore:.4f}", log_file, silent)
    cm = confusion_matrix(y_test, y_pred)
    utils.log(f"    conf. matrix:\n{cm}", log_file, silent)
    plot_cm(cm)

    def plot_pca(X, y, name):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)  # Reduce to 2D

        colors = ['blue' if label == 0 else 'orange' for label in y]

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], color=colors, alpha=0.7)
        blue_patch = plt.Line2D([0], [0], marker='o', color='w',
                                markerfacecolor='blue', markersize=10,
                                label="PU = 0")
        orange_patch = plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='orange', markersize=10,
                                  label="PU = 1")
        plt.legend(handles=[blue_patch, orange_patch], loc='upper right')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5,
                 color='gray')

        plt.title(f"PCA 2D projection of {name}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        pca_plot_path = utils.path_join(out_dir, f"pca_2d_{name}.png")
        plt.savefig(pca_plot_path)
        plt.close()

        print(f"PCA 2D projection for {name} saved to {pca_plot_path}")
    # 2D pca proyection y_true vs y_pred
    plot_pca(X_test, y_test, "train_true")
    plot_pca(X_test, y_pred, "train_pred")


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2")
    parser.add_argument('train_csv', type=str, nargs='*',
                        help="Path(s) to the train CSV file(s)")
    parser.add_argument('-t', '--test-csv', type=str, required=True,
                        help="Path to the train CSV file")
    parser.add_argument('-p', '--params-path', type=str, required=True,
                        help="Path to the JSON file with params")
    parser.add_argument('-c', '--classifier', choices=['rforest', 'stacking'],
                        required=True,
                        help="Train either 'rforest' or 'stacking' classifier")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Output directory to save results")
    parser.add_argument('-s', '--silent', action='store_true',
                        help="Silent output")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    print("rq2: loading params")
    params = utils.json_load(args.params_path)

    print("rq2: loading data")
    X_train, y_train = train.merge_data(args.train_csv)
    X_test, y_test = utils.load_data(args.test_csv)

    if args.classifier == "rforest":
        print("rq2: training 'RandomForestClassifier'")
        cls, pca = train.train_RandomForestClassifier(X_train, y_train, params)
    elif args.classifier == "stacking":
        print("rq2: training 'StackingClassifier'")
        cls, pca = train.train_StackingClassifier(X_train, y_train, params)
    else:
        print("rq2: Unknown classifier")
        exit(1)

    if pca is not None:
        print("rq2: PCA fit")
        X_test = pca.transform(X_test)

    test(cls, X_test, y_test, args.output_dir, silent=args.silent)
