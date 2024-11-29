import argparse
import time

from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

import utils
import train


def test(X_train, y_train, X_test, y_test,
         params_rf, params_sv, params_gr, params_st, out_dir, silent=False):
    '''
    Evaluates multiple classifiers on a given dataset, including training,
    prediction, and evaluation phases. Results are saved.
    1. Training:
        - Trains the following classifiers:
          - RandomForestClassifier
          - SVC
          - GradientBoostingClassifier
          - StackingClassifier
    2. Data Transformation:
        - Applies PCA and scaling (if available) to test data for consistency
          with the training pipeline.
    3. Prediction:
        - Makes predictions using each trained classifier.
        - Implements a custom stacking method (`StackMax`) that aggregates
          predictions using majority voting.
        - Logs prediction time for each classifier.
    4. Evaluation:
        - Computes and logs metrics for each classifier:
          - Accuracy
          - F1 Score for PU = 0
          - F1 Score for PU = 1
          - Macro F1 Score
          - Weighted F1 Score
    5. **Visualization**:
        - Creates a combined bar chart comparing classifier metrics.
        - Saves the chart as a PNG file in the output directory.

    :param X_train : array-like of shape (n_samples, n_features)
    :param y_train : array-like of shape (n_samples,)
    :param X_test : array-like of shape (n_samples, n_features)
    :param y_test : array-like of shape (n_samples,)
    :param params_rf : dict Hyperparameters for the RandomForestClassifier.
    :param params_sv : dict Hyperparameters for the SVC.
    :param params_gr : dict Hyperparameters for the GradientBoostingClassifier.
    :param params_st : dict Hyperparameters for the StackingClassifier.
    :param out_dir : str Directory where results and plots will be saved.
    :param silent : bool, suppress some output (def.=False).
    '''
    utils.mkdir(out_dir)
    log_file = utils.path_join(out_dir, "results.txt")
    utils.log_init(log_file)

    # train ###################################################################
    utils.log("rq2: training 'RandomForestClassifier'", log_file, silent)
    start_time = time.time()
    cls_rf, pca_rf, sc_rf = train.train_RandomForestClassifier(X_train, y_train,
                                                               params_rf)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    utils.log("rq2: training 'SVC'", log_file, silent)
    start_time = time.time()
    sv_cls, sv_pca, sv_sc = train.train_SVC(X_train, y_train, params_sv)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    utils.log("rq2: training 'GradientBoostingClassifier'", log_file, silent)
    start_time = time.time()
    cls_gr, pca_gr, sc_gr = train.train_GradientBoostingClassifier(X_train,
                                                                   y_train,
                                                                   params_gr)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    utils.log("rq2: training 'StackingClassifier'", log_file, silent)
    start_time = time.time()
    cls_st, pca_st, sc_st = train.train_StackingClassifier(X_train, y_train,
                                                           params_st)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    # PAC + scaler ############################################################
    utils.log("rq2: setting up test data", log_file, silent)
    start_time = time.time()
    if pca_rf is not None:
        if sc_rf is not None:
            print("    PCA + scaler transform rf")
            X_test_rf = pca_rf.transform(X_test)
            X_test_rf = sc_rf.transform(X_test_rf)
        else:
            print("    PCA transform rf")
            X_test_rf = pca_rf.transform(X_test)
    else:
        X_test_rf = X_test
    if sv_pca is not None:
        if sv_sc is not None:
            print("    PCA + scaler transform sv")
            X_test_sv = sv_pca.transform(X_test)
            X_test_sv = sv_sc.transform(X_test_sv)
        else:
            print("    PCA transform svc")
            X_test_sv = sv_pca.transform(X_test)
    else:
        X_test_sv = X_test
    if pca_gr is not None:
        if sc_gr is not None:
            print("    PCA + scaler transform gr")
            X_test_gr = pca_gr.transform(X_test)
            X_test_gr = sc_gr.transform(X_test_gr)
        else:
            print("    PCA transform gr")
            X_test_gr = pca_gr.transform(X_test)
    else:
        X_test_gr = X_test
    if pca_st is not None:
        if sc_st is not None:
            print("    PCA + scaler transform st")
            X_test_st = pca_st.transform(X_test)
            X_test_st = sc_st.transform(X_test_st)
        else:
            print("    PCA transform st")
            X_test_st = pca_st.transform(X_test)
    else:
        X_test_st = X_test

    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    # predictions #############################################################
    utils.log("rq2: making predictions", log_file, silent)

    utils.log("    Predicting with RandomForestClassifier", log_file, silent)
    start_time = time.time()
    y_pred_rf = cls_rf.predict(X_test_rf)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    utils.log("    Predicting with SVC", log_file, silent)
    start_time = time.time()
    y_pred_sv = sv_cls.predict(X_test_sv)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    utils.log("    Predicting with GradientBoostingClassifier", log_file,
              silent)
    start_time = time.time()
    y_pred_gr = cls_gr.predict(X_test_gr)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    utils.log("    Predicting with StackingClassifier", log_file, silent)
    start_time = time.time()
    y_pred_st = cls_st.predict(X_test_st)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    # StackMax logic: Use the mode of predictions across classifiers
    utils.log("    Predicting with StackMax (custom stacking)",
              log_file, silent)
    stacked_predictions = np.vstack([y_pred_st, y_pred_gr, y_pred_rf])
    start_time = time.time()
    y_pred_stackmax = np.apply_along_axis(lambda x: np.bincount(x).argmax(),
                                          axis=0, arr=stacked_predictions)
    elapsed_time = time.time() - start_time
    utils.log(f"    time: {elapsed_time:.2f} seconds", log_file, silent)

    # eval ####################################################################
    # Evaluate and compare classifiers
    utils.log("rq2: evaluating classifiers", log_file, silent)

    metrics = {
        "Accuracy": [],
        "F1 Score (PU=0)": [],
        "F1 Score (PU=1)": [],
        "F1 Score (Macro)": [],
        "F1 Score (Weighted)": [],
    }
    classifiers = {
        "RandomForest": y_pred_rf,
        "SVC": y_pred_sv,
        "GradientBoosting": y_pred_gr,
        "Stacking": y_pred_st,
        "StackMax": y_pred_stackmax,
    }

    classifier_names = list(classifiers.keys())

    for name, preds in classifiers.items():
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        f1_weighted = f1_score(y_test, preds, average="weighted")

        f1_PU_0 = f1_score(y_test, preds, labels=[0], average=None)[0]
        f1_PU_1 = f1_score(y_test, preds, labels=[1], average=None)[0]

        metrics["Accuracy"].append(acc)
        metrics["F1 Score (PU=0)"].append(f1_PU_0)
        metrics["F1 Score (PU=1)"].append(f1_PU_1)
        metrics["F1 Score (Macro)"].append(f1_macro)
        metrics["F1 Score (Weighted)"].append(f1_weighted)

        # Log the performance
        utils.log(f"{name} Performance:", log_file, silent)
        utils.log(f"    Accuracy: {acc:.6f}", log_file, silent)
        utils.log(f"    F1 Score (PU=0): {f1_PU_0:.6f}", log_file, silent)
        utils.log(f"    F1 Score (PU=1): {f1_PU_1:.6f}", log_file, silent)
        utils.log(f"    F1 Score (Macro): {f1_macro:.6f}", log_file, silent)
        utils.log(f"    F1 Score (Weighted): {f1_weighted:.5f}",
                  log_file, silent)

    # Plot combined bar chart
    metric_names = list(metrics.keys())
    num_metrics = len(metric_names)
    num_classifiers = len(classifier_names)

    # Data for plotting
    bar_width = 0.12  # Width of each bar
    x = np.arange(num_metrics)  # Positions for each metric on the X-axis

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each classifier's metrics
    for i, classifier_name in enumerate(classifier_names):
        offset = (i - (num_classifiers - 1) / 2) * bar_width
        ax.bar(x + offset, [metrics[metric][i] for metric in metric_names],
               bar_width, label=classifier_name)

    # Customize the chart
    ax.set_title("Classifier Comparison")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)  # Assuming metrics are normalized (0-1)
    ax.legend(title="Classifiers")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(utils.path_join(out_dir, "classifier_comparison_combined.png"))

    utils.log("rq2: evaluation and combined plotting complete",
              log_file, silent)


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2")
    parser.add_argument('train_csv', type=str, nargs='*',
                        help="Path(s) to the train CSV file(s)")
    parser.add_argument('-t', '--test-csv', type=str, required=True,
                        help="Path to the train CSV file")
    parser.add_argument('-A', '--rforest-params-path', type=str, required=True,
                        help="Path to the JSON file with the rforest params")
    parser.add_argument('-B', '--gradient-params-path', type=str, required=True,
                        help="Path to the JSON file with the gradient params")
    parser.add_argument('-C', '--svc-params-path', type=str, required=True,
                        help="Path to the JSON file with the svc params")
    parser.add_argument('-D', '--stacking-params-path', type=str, required=True,
                        help="Path to the JSON file with the stacking params")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Output directory to save results")
    parser.add_argument('-s', '--silent', action='store_true',
                        help="Silent output")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    print("rq2: loading params")
    params_rf = utils.json_load(args.rforest_params_path)
    params_sv = utils.json_load(args.svc_params_path)
    params_gr = utils.json_load(args.gradient_params_path)
    params_st = utils.json_load(args.stacking_params_path)

    print("rq2: loading data")
    X_train, y_train = train.merge_data(args.train_csv)
    X_test, y_test = utils.load_data(args.test_csv)

    test(X_train, y_train, X_test, y_test,
         params_rf, params_sv, params_gr, params_st,
         args.output_dir, silent=args.silent)

