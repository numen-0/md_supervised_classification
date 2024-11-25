import argparse
import time

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import utils


def test(train_path, dev_path, out_dir, silent=False):
    '''
    1. load data.
    2. get metadata size, number of instances, and dimensions.
    3. train a classifier (Random Forest).
    4. eval the model on F1-score and confusion matrix.

    :param train_path (str): path to the training CSV file.
    :param dev_path (str):   path to the development CSV file.
    :param out_dir (str):    output directory where results will be saved.
    :param silent (bool):    suppress log output (def.: False).
    '''
    utils.mkdir(out_dir)
    log_file = utils.path_join(out_dir, "results.txt")
    utils.log_init(log_file)

    # step 1: load data
    utils.log("rq1: loading data", log_file, silent)
    start_time = time.time()
    X_train, y_train = utils.load_data(train_path)
    X_dev, y_dev = utils.load_data(dev_path)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    # step 2: meta information
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

        plt.title(f"PCA 2D projection of {name} set")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        pca_plot_path = utils.path_join(out_dir, f"pca_2d_{name}.png")
        plt.savefig(pca_plot_path)
        plt.close()

        print(f"PCA 2D projection for {name} saved to {pca_plot_path}")

    def dataset_meta(X, y, name):
        size_mb = (X.nbytes + y.nbytes) / (1024 * 1024)
        num_instances, num_dimensions = X.shape
        utils.log(f"rq1:meta_data: {name}", log_file, silent)
        utils.log(f"    size: {size_mb:.2f}MB", log_file, silent)
        utils.log(f"    instances: {num_instances}", log_file, silent)
        utils.log(f"    dimensions: {num_dimensions}", log_file, silent)
        plot_pca(X, y, name)

    dataset_meta(X_train, y_train, "train")
    dataset_meta(X_dev, y_dev, "dev")

    # step 3: train classifier
    utils.log("rq2: training model", log_file, silent)
    start_time = time.time()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    utils.log(f"    training time: {
              elapsed_time:.2f} seconds", log_file, silent)

    # step 4: evaluate model
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

    utils.log("rq3: evaluating model", log_file, silent)
    y_pred = clf.predict(X_dev)
    fscore = classification_report(y_dev, y_pred, labels=[0, 1])
    utils.log(f"{fscore}", log_file, silent)
    cm = confusion_matrix(y_dev, y_pred)
    utils.log(f"    conf. matrix:\n{cm}", log_file, silent)
    plot_cm(cm)


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ1")
    parser.add_argument('-t', '--train-csv', type=str, required=True,
                        help="Path to the train CSV file")
    parser.add_argument('-d', '--dev-csv', type=str, required=True,
                        help="Path to the dev CSV file")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Output directory to save results")
    parser.add_argument('-s', '--silent', action='store_true',
                        help="Silent output")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    test(args.train_csv, args.dev_csv, args.output_dir, silent=args.silent)
