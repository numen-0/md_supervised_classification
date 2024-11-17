from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import utils


def analyze_raw(data, output_dir, prefix="raw_data", silent=False):
    """
    Analyze from raw data (UI, PCM, RS, PU, TXT), PU, UI and TXT

    works better after filtering unwanted lines, and correcting the PU

    all (text and plots) will be saved to @output_dir@/@prefix@_file_name.extension

    :param data: DataFrame
    :param output_dir: Directory to save outputs
    :param prefix: Prefix for the output files
    :param silent: If True, suppresses print statements
    """
    utils.mkdir(output_dir)
    prefix = utils.path_join(output_dir, prefix)
    log_file = prefix + '_analysis.txt'

    utils.log(f"analyzer: analyzing raw data (saving to '{log_file}')", log_file, silent)
    utils.log_init(log_file)

    # vars
    total_instances = len(data)
    total_unique_uis = data['UI'].nunique()
    k_list = [4, 7, 9]
    number_of_words = 5

    # PU
    utils.log(f"[PU]:", log_file, silent)
    pu_distribution = data.groupby('PU')['UI'].nunique() / total_unique_uis * 100
    utils.log(f"\tPU class distribution:\n{pu_distribution}", log_file, silent)
    # plot
    fig, ax = plt.subplots()
    sns.barplot(x=pu_distribution.index, y=pu_distribution.values, ax=ax)
    ax.set_title('PU Class Distribution')
    ax.set_xlabel('PU')
    ax.set_ylabel('Percentage (%)')
    utils.fig_save(fig, prefix + '_pu_class_distribution.png')

    # UI
    utils.log(f"[UI]:", log_file, silent)
    unique_uis = data['UI'].nunique()
    mean_instances_per_ui = total_instances / unique_uis if unique_uis else 0

    utils.log(f"\ttotal number of instances: {total_instances}", log_file, silent)
    utils.log(f"\tnumber of unique UIs: {unique_uis}", log_file, silent)
    utils.log(f"\tmean number of instances per UI: {mean_instances_per_ui:.2f}", log_file, silent)

    # TXT - length
    utils.log(f"[TXT]:length", log_file, silent)
    data['text_length'] = data['TXT'].apply(len)
    text_length_stats = data['text_length'].describe()
    utils.log(f"\ttext length statistics:\n{text_length_stats}", log_file, silent)
    # plot
    fig, ax = plt.subplots()
    sns.histplot(data['text_length'], bins=20, kde=True, ax=ax)
    ax.set_title('Text Length Distribution')
    ax.set_xlabel('Text Length (characters)')
    ax.set_ylabel('Frequency')
    utils.fig_save(fig, prefix + '_text_length_distribution.png')

    # # TXT - words
    utils.log(f"[TXT]:words", log_file, silent)
    for pu_value in [0, 1]:
        utils.log(f"analyzing words for PU = {pu_value}:", log_file, silent)
        filtered_data = data[data['PU'] == pu_value]
        all_words = ' '.join(filtered_data['TXT']).lower().split()

        for k in k_list:
            filtered_words = [word for word in all_words if len(word) > k]

            word_count = Counter(filtered_words)
            most_common_words = word_count.most_common(number_of_words)
            utils.log(f"\tmost common words len(word) > {k}:", log_file, silent)
            for word, count in most_common_words:
                utils.log(f"\t\t{word:>12}: {count}", log_file, silent)  # Aligns words to the right

    utils.log("analyzer: raw analysis completed", log_file, silent)


def analyze_ext(data, output_dir, prefix="raw_data", silent=False):
    """
    Analyze ext data (PU, F00, F01, F10, F11)
    :param data: DataFrame
    :param output_dir: Directory to save outputs
    :param prefix: Prefix for the output files
    :param silent: If True, suppresses print statements
    """
    utils.mkdir(output_dir)
    prefix = utils.path_join(output_dir, prefix)
    log_file = prefix + '_analysis_ext.txt'

    utils.log(f"analyzer: analyzing ext data (saving to '{log_file}')", log_file, silent)
    utils.log_init(log_file)

    ext_cols = ['F00', 'F01', 'F10', 'F11']

    normalized_distribution = pd.DataFrame(index=ext_cols, columns=[0, 1])

    for ext in ext_cols:
        ext_vs_pu = data.groupby(['PU'])[ext].mean() * 100
        normalized_distribution.loc[ext, 0] = ext_vs_pu.get(0, 0)
        normalized_distribution.loc[ext, 1] = ext_vs_pu.get(1, 0)

    utils.log(f"[EXT]:", log_file, silent)
    utils.log(f"\tFXX vs PU normalized distribution:\n{normalized_distribution}", log_file, silent)
    # plot
    fig, ax = plt.subplots()
    sns.heatmap(normalized_distribution.astype(float), annot=True, cmap='Greens', ax=ax,
                cbar_kws={'label': 'Percentage (%)'})
    ax.set_title('Normalized Distribution of FXX vs PU')
    ax.set_xlabel('PU')
    ax.set_ylabel('FXX')
    utils.fig_save(fig, prefix + '_ext_vs_pu_heatmap.png')

    utils.log("analyzer: ext analysis completed", log_file, silent)
