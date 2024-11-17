from sys import exit
import argparse
import re

import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

import utils
import analyzer


###############################################################################
## globals ####################################################################
defaults = {
    "safe":            False,
    "chop":            0.0,
    "split":           [0.9, 0.0, 0.1],
    "split_and_parse": False,
    "silent_analysis": False,
    "skip_analysis":   False,
}
config = {
    "paths":            [],
    "output_dir":       None,
    "safe":             False,
    "chop":             0.0,
    "split":            [0.9, 0.0, 0.1],
    "split_and_parse":  False,
    "silent_analysis":  False,
    "skip_analysis":    False,
    "analyzer_dir":     None,
}


###############################################################################
## api ########################################################################
def data_filter_out(data):
    """
    Filter out invalid rows based on specific column criteria:

    UI:        E[-1-9]+
    PCM,RS,PU: [01]
    TXT:       string

    :param data: DataFrame
    :return: Filtered DataFrame
    """
    ui_pattern = r"^E[0-9]+$"

    valid_ui = data['UI'].apply(lambda x: bool(re.match(ui_pattern, str(x))))
    valid_pcm = data['PCM'].isin(['0', '1'])
    valid_rs = data['RS'].isin(['0', '1'])
    valid_pu = data['PU'].isin(['0', '1'])
    valid_txt = data['TXT'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)

    valid_rows = valid_ui & valid_pcm & valid_rs & valid_pu & valid_txt

    return data[valid_rows].reset_index(drop=True)


def data_OR_logic_PU(data):
    """
    Apply OR-logic to the PU column, spreading the combined value across the same UI rows
    :param data: DataFrame
    :return: DataFrame
    """
    def _or_logic(series):
        return reduce(lambda x, y: x | y, series)

    or_class_per_ui = data.groupby('UI')['PU'].apply(_or_logic)

    # Map the OR result back to the original DataFrame by UI
    data['PU'] = data['UI'].map(or_class_per_ui)

    return data


def _count_and_normalize(group):
    """
    Count and normalize 'PCM_RS' values in the group.

    :param group: DataFrame; Input group containing a 'PCM_RS' column.
    """
    total_count = len(group)
    counts = group['PCM_RS'].value_counts()

    normalized_counts = [
        counts.get('00', 0) / total_count,  # F00
        counts.get('01', 0) / total_count,  # F01
        counts.get('10', 0) / total_count,  # F10
        counts.get('11', 0) / total_count   # F11
    ]
    return pd.Series(normalized_counts, index=['F00', 'F01', 'F10', 'F11'])


def data_combine_UI_safe(data):
    """
    Group the rows by both 'UI' and 'PU', apply necessary aggregation,
    and normalize PCM/RS combinations (00, 01, 10, 11), while ensuring
    that the combined TXT does not exceed the token limit.
    PU or logic must be done before this.

    :param data: DataFrame
    :return: DataFrame grouped by 'UI' and 'PU' with normalized combinations.
    """
    # we import here because it's very heavy to load...
    from tokenization import validate_tokens
    combined_rows = []

    # Iterating over each group of UI and PU
    for (ui, pu), group in data.groupby(['UI', 'PU']):
        current_txt = []
        current_row = {'UI': ui, 'PU': pu, 'TXT': ''}
        for _, row in group.iterrows():
            joined_txt = ' '.join(current_txt + [row['TXT']])

            # Check if the new TXT is valid according to tokenization rules
            if validate_tokens(joined_txt): # update the current TXT
                current_txt.append(row['TXT'])
            else:                           # save current row and start a new
                current_row['TXT'] = ' '.join(current_txt)
                combined_rows.append(current_row)

                current_txt = [row['TXT']]
                current_row = {'UI': ui, 'PU': pu, 'TXT': ''}

        if current_txt:
            current_row['TXT'] = ' '.join(current_txt)
            combined_rows.append(current_row)

    # Create a new DataFrame from the combined rows
    grouped_data = pd.DataFrame(combined_rows)

    # Combine PCM and RS
    data['PCM_RS'] = data['PCM'].astype(str) + data['RS'].astype(str)

    normalized_data = data.groupby('UI', group_keys=False).apply(_count_and_normalize).reset_index()

    final_data = pd.merge(grouped_data, normalized_data, on='UI', how='inner')

    return final_data


def data_combine_UI_unsafe(data):
    """
    Group the rows by both 'UI' and 'PU', apply necessary aggregation,
    and normalize PCM/RS combinations (00, 01, 10, 11). PU or logic must
    be done before this.

    :param data: DataFrame
    :return: DataFrame grouped by 'UI' and 'PU' with normalized combinations.
    """
    grouped_data = data.groupby(['UI', 'PU']).agg({
        'TXT': ' '.join,
    }).reset_index()

    # combine PCM and RS
    data['PCM_RS'] = data['PCM'].astype(str) + data['RS'].astype(str)

    normalized_data = data.groupby('UI', group_keys=False).apply(_count_and_normalize,
                                                                 include_groups=False).reset_index()

    final_data = pd.merge(grouped_data, normalized_data, on='UI', how='inner')

    return final_data


def data_filter_rare_classes(data, min_count=3):
    """
    Filter out rows with rare 'PU' values.

    :param data:      DataFrame containing the 'PU' column
    :param min_count: Minimum count of instances required for a class to be retained
    :return: DataFrame with rare classes filtered out
    """
    class_counts = data['PU'].value_counts()

    valid_classes = class_counts[class_counts >= min_count].index
    filtered_data = data[data['PU'].isin(valid_classes)].copy()

    return filtered_data.reset_index(drop=True)


def data_split_2(data, split, stratify=None):
    """
    split the DataFrame into 2 sets.
        size(data_0) = size(data) * (1 - split)
        size(data_1) = size(data) * split
    :param data:             DataFrame;
    :param split:            float
    :param stratify: list|None;
    :return: data_0, data_1
    """
    if split == 0.0:
        return data, data.iloc[0:0]

    stratify_data = data[stratify].values if stratify else None
    return train_test_split(data, test_size=split, stratify=stratify_data)


def data_split_3(data, split, stratify=None):
    """
    split the DataFrame into train, dev and test  sets.
    :param data:             DataFrame;
    :param split:            list;     [train, dev, test]
    :param stratify: list|None;
    :return: train_df, dev_df, test_df
    """
    _, dev_size, test_size = split

    if test_size + dev_size == 0.0:
        return data, data.iloc[0:0], data.iloc[0:0]

    train_df, temp_df = data_split_2(data, test_size + dev_size, stratify)

    if dev_size == 0.0:
        return train_df, data.iloc[0:0], temp_df
    if test_size == 0.0:
        return train_df, temp_df, data.iloc[0:0]

    test_size = test_size / (dev_size + test_size)  # scale the size
    dev_df, test_df = data_split_2(temp_df, test_size, stratify)

    return train_df, dev_df, test_df


def data_parse(data, analyze=False, safe=False):
    """
    parse the DataFrame
    :param data:  DataFrame;
    :param analyze: analyze the data or not
    :param safe: check token lengths
    :return:      parsed_data
    """
    print(f"\tfiltering out invalid rows")
    data = data_filter_out(data)

    if safe:
        data_combine_UI = data_combine_UI_safe
    else:
        data_combine_UI = data_combine_UI_unsafe

    data['PU'] = data['PU'].astype(int)
    data['PCM'] = data['PCM'].astype(int)
    data['RS'] = data['RS'].astype(int)

    print(f"\tspreading PU value, across UI")
    data = data_OR_logic_PU(data)

    if config["chop"] > 0.0:
        print(f"\tcutting data")
        data, _ = data_split_2(data, config["chop"], stratify=["PU"])

    # NOTE: I prefer to perform this analysis after chopping the data so that we
    #       obtain insights about the current data (before split) rather than
    #       relying on any remnants we may have removed earlier.
    data_raw = pd.DataFrame()
    data_fxx = pd.DataFrame()
    if analyze:
        data_raw = data.copy()

    if config["split_and_parse"]:
        print(f"\tsplitting data")
        train_df, dev_df, test_df = data_split_3(data, config["split"],
                                                 stratify=["PU"])
        print(f"\tcombining UI instances")
        train_df = data_combine_UI(train_df)
        test_df = data_combine_UI(test_df)
        dev_df = data_combine_UI(dev_df)

        if analyze:
            data_fxx = pd.concat([train_df, test_df, dev_df], ignore_index=True)
    else:
        print(f"\tcombining UI instances")
        data = data_combine_UI(data)
        print(f"\tsplitting data")
        train_df, dev_df, test_df = data_split_3(data, config["split"],
                                                 stratify=["PU"])
        if analyze:
            data_fxx = data

    if analyze:
        print(f"preprocess: meta data")
        print(f"\traw   size: {len(data)}")
        print(f"\ttrain size: {len(train_df)}\t({config['split'][0] * 100}%)")
        print(f"\tdev   size: {len(dev_df)}\t({config['split'][1] * 100}%)")
        print(f"\ttest  size: {len(test_df)}\t({config['split'][2] * 100}%)")

        analyzer.analyze_raw(data_raw, config["analyzer_dir"], prefix="post_filter",
                             silent=config["silent_analysis"])
        analyzer.analyze_ext(data_fxx, config["analyzer_dir"], prefix="post_filter",
                             silent=config["silent_analysis"])

    return train_df, dev_df, test_df


def data_reparse(conf_file):
    """
    Reparse and reprocess the configuration, in safe mode.

    :param conf_file: str; Path to the configuration file.
    """
    print(f"preprocess:reparse: reparsing")
    load_setup(conf_file)
    config["silent_analysis"] = True
    config["safe"] = True
    filename = utils.path_join(config["output_dir"], "preprocess_conf.json")
    utils.json_save(config, filename)

    execute()


def data_save(data, name):
    """
    Save subsets of the DataFrame as CSV files.

    :param data: DataFrame; Input data containing 'PU', 'TXT', 'F00', 'F01', 'F10', 'F11', and 'UI'.
    :param name: str; Base name for the output files.

    - Saves 'PU' and 'TXT' columns as `{name}.csv`.
    - Saves 'PU', 'F00', 'F01', 'F10', 'F11' as `extern/{name}.csv`.
    - Saves 'UI' and an 'INDX' as `links/{name}.csv`.
    """
    data['INDX'] = range(len(data))
    pu_txt_df = data[['PU', 'TXT']]

    pu_ext_df = data[['PU', 'F00', 'F01', 'F10', 'F11']]

    links_df = data[['UI', 'INDX']]

    utils.csv_save(pu_txt_df, f"{
        utils.path_join(config['output_dir'], name)}.csv")
    utils.csv_save(pu_ext_df, f"{
        utils.path_join(config['output_dir'], "extern/" + name)}.csv")
    utils.csv_save(links_df, f"{
        utils.path_join(config['output_dir'], "links/" + name)}.csv")


def setup(paths, output_dir, chop=0.0, split=(0.9, 0.0, 0.1),
          split_and_parse=False, silent_analysis=False, skip_analysis=False,
          safe=False):
    """
    Configure preprocessing setup.

    :param paths: list[str]; Input file paths.
    :param output_dir: str; Directory for saving output files.
    :param chop: float; Fraction of data to discard (default: 0.0).
    :param split: tuple; Proportions for train/dev/test split (default: (0.9, 0.0, 0.1)).
    :param split_and_parse: bool; If True, split data and parse (default: False).
    :param silent_analysis: bool; If True, suppress output during analysis (default: False).
    :param skip_analysis: bool; If True, skip the analysis phase (default: False).
    :param safe: bool; If True, operate in safe mode (default: False).
    """
    config["paths"] = paths
    config["split_and_parse"] = split_and_parse
    config["output_dir"] = output_dir
    config["chop"] = chop
    config["safe"] = safe
    config["skip_analysis"] = skip_analysis
    config["silent_analysis"] = silent_analysis
    config["analyzer_dir"] = utils.path_join(output_dir, "meta")

    while len(split) < 3:
        split.append(0)  # ensure len == 3
    if sum(split) != 1.0:
        print("preprocess:setup: split values must add to 100")
        exit(1)
    config["split"] = split

    print(f"preprocess:setup: saving setup")
    utils.mkdir(config["output_dir"])
    filename = utils.path_join(config["output_dir"], "preprocess_conf.json")
    utils.json_save(config, filename)

    print(f"preprocess:setup: done")
    print(f"\tpath(s):         {config['paths']}")
    print(f"\toutput_dir:      {config['output_dir']}")
    print(f"\tsafe:            {config['safe']}")
    print(f"\tchop:            {config['chop']}")
    print(f"\tsplit:           {config['split']}")
    print(f"\tsplit-and-parse: {config['split_and_parse']}")
    print(f"\tsilent_analysis: {config['silent_analysis']}")
    print(f"\tskip-analysis:   {config['skip_analysis']}")
    print(f"\tanalyzer_dir:    {config['analyzer_dir']}")


def load_setup(path):
    """
    Load and apply the preprocessing setup from a configuration file.

    :param path: str; Path to the JSON config file.

    - Loads the configuration settings from the specified file.
    - If loading fails, prints an error and exits.
    - Passes the loaded settings to 'setup()' to configure the preprocessing.
    """
    loaded_config = utils.json_load(path)
    if loaded_config is None:
        print(f"preprocess:load_setup: could not load setup from {path}")
        exit(1)
    setup(loaded_config["paths"], loaded_config["output_dir"],
          chop=loaded_config.get("chop", defaults["chop"]),
          split=loaded_config.get("split", defaults["split"]),
          split_and_parse=loaded_config.get("split_and_parse", defaults['split_and_parse']),
          silent_analysis=loaded_config.get("silent_analysis", defaults['silent_analysis']),
          skip_analysis=loaded_config.get("skip_analysis", defaults['skip_analysis']),
          safe=loaded_config.get("safe", defaults["safe"]))


def init():
    """
    Initialize preprocessing by parsing command-line arguments.

    - Loads settings from a configuration file if '--config' is provided.
    - If '--config' is not used, 'csv_path' and 'output_dir' must be specified.

    If a config file is not provided, it sets up the process using the provided CSV paths and output directory.
    """
    parser = argparse.ArgumentParser(description="preprocess CSV file(s)")
    # args
    parser.add_argument('--config', type=str,
                        help="Path to config.json file to load custom settings")
    parser.add_argument('csv_path', type=str, nargs='*',
                        help="Path(s) to the CSV file(s)")
    parser.add_argument('-o', '--output-dir', type=str,
                        help="Output directory to save the processed data")
    parser.add_argument('-s', '--split', type=str,
                        help="Split percentages in the format train/dev/test (e.g., 80/10/10, 80//20); def: 90//10")
    parser.add_argument('-c', '--chop', type=int,
                        help="Chop out percentage of instances (e.g., 25); def: 0")
    parser.add_argument('--split-and-parse', action='store_true',
                        help="First split and then parse CSV(s), if not set it will first parse, then split the data")
    parser.add_argument('--skip-analysis', action='store_true',
                        help="Skip analysis")
    parser.add_argument('--silent-analysis', action='store_true',
                        help="Silence analyzer output")
    parser.add_argument('--safe', action='store_true',
                        help="Check if instances TXT fits our tokenization, and split instances if necessary")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    if args.config is not None:  # load config from file
        load_setup(args.config)
        return

    if not args.csv_path or not args.output_dir:
        print("Error: If --config is not provided, --csv_path and --output-dir must be specified.")
        parser.print_help()
        exit(1)

    # save args
    if args.split is not None:  # train/test/dev split
        split = args.split.split('/')
        split = [int(part) / 100 if part else 0 for part in split]
    else:
        split = defaults['split']

    if args.chop is not None:  # chop
        chop = int(args.chop) / 100
    else:
        chop = defaults['chop']

    setup(args.csv_path, args.output_dir, split_and_parse=args.split_and_parse,
          chop=chop, split=split, silent_analysis=args.silent_analysis,
          skip_analysis=args.skip_analysis, safe=args.safe)


"""
NOTE: I used this cool cmd to check if the preprocess is doing a great job it
      sould give the same number of instances in default config. cool stuff:

grep -oE "^E[0-9]+,[01],[01],[01]," data/raw/DataI_MD.csv | cut -d , -f1 | sort -u |wc -l
 -> 712

wc -l data/processed/*.csv
 -> 143 data/processed/dev.csv
 ->  73 data/processed/test.csv
 -> 499 data/processed/train.csv
 -> 715 total

    (it gives +3 lines because of the csv headers)
"""
def execute():
    """
    Execute the preprocessing workflow.

    - Loads data from CSV file(s) specified in 'config["paths"]'.
    - Concatenates the data into a single DataFrame.
    - Parses and splits the data into train, dev, and test sets.
    - Saves the split datasets to the output directory:
        - 'train.csv', 'dev.csv', and 'test.csv' are saved in the output directory.
        - Additional subdirectories 'extern' and 'links' are created for external data and link files.
    """
    # load data
    dataframes = []
    print(f"preprocess: loading data")
    for path in config["paths"]:
        data = utils.csv_load(path)
        if data is None:
            exit(1)
        dataframes.append(data)

    data = pd.concat(dataframes, ignore_index=True)

    # modify and split the data
    print(f"preprocess: parsing data")
    train_df, dev_df , test_df= data_parse(data, analyze=True, safe=config["safe"])

    # save data
    print(f"preprocess: saving data to {config['output_dir']}")
    utils.mkdir(config["output_dir"])
    utils.mkdir(utils.path_join(config['output_dir'], "extern"))
    utils.mkdir(utils.path_join(config['output_dir'], "links"))

    if config["split"][0] > 0.0:
        data_save(train_df, "train")
    if config["split"][1] > 0.0:
        data_save(dev_df, "dev")
    if config["split"][2] > 0.0:
        data_save(test_df, "test")

    print(f"preprocess: done")


###############################################################################
## main #######################################################################
if __name__ == "__main__":
    init()
    execute()
