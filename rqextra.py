
import argparse

from sklearn.metrics import f1_score
import numpy as np

import utils


def test(X_test, y_test, out_file, silent=False):
    '''
    Get baselines with zero-rule and random cls

    :param X_test: Test feature data
    :param y_test: True labels for test set
    :param out_file: Path to the log/output file
    :param silent: Suppress console output if True
    '''
    utils.mkdir(utils.path_dirname(out_file))
    log_file = out_file
    utils.log_init(log_file)

    # Random baseline
    print("rqextra: random baseline")
    random_f1_scores = []
    for _ in range(10):
        y_random = np.random.choice(np.unique(y_test), size=len(y_test))
        random_f1_scores.append(f1_score(y_test, y_random, average='weighted'))
    random_f1_mean = np.mean(random_f1_scores)
    random_f1_std = np.std(random_f1_scores)

    utils.log(f"Random Baseline - F1 Score (Mean, Std): "
              f"{random_f1_mean:.4f}, {random_f1_std:.4f}", log_file, silent)

    # Zero-rule baseline
    print("rqextra: zero-rule baseline")
    majority_class = np.argmax(np.bincount(y_test))
    y_zero = [majority_class] * len(y_test)
    zero_rule_f1 = f1_score(y_test, y_zero, average='weighted')

    utils.log(f"Zero-Rule Baseline - F1 Score: {zero_rule_f1:.4f}", log_file,
              silent)

    print("rqextra: evaluation completed")


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2")
    parser.add_argument('-t', '--test-csv', type=str, required=True,
                        help="Path to the train CSV file")
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help="Output file to save results")
    parser.add_argument('-s', '--silent', action='store_true',
                        help="Silent output")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    print("rqextra: loading data")
    X_test, y_test = utils.load_data(args.test_csv)

    test(X_test, y_test, args.output_file, silent=args.silent)
