import argparse

import numpy as np
import spacy
import pandas as pd

import utils


def vectorize_bow(path, output_file):
    # TODO:
    pass


def vectorize_spacy(path, output_file, batch_size=128):
    """
    Vectorize the input text using SpaCy.
    :param path: Path to the CSV file
    :param output_file: File to save the output CSV
    :param batch_size: The size of the batches
    """
    print("vectorization: vectorizing using SpaCy", flush=True)
    sentences = utils.csv_load(path)
    texts = sentences['TXT'].values
    nlp = spacy.load("es_core_news_md")

    print("vectorization:vectorize_spacy: batching", flush=True)
    docs = []
    for i in range(0, len(texts), batch_size):
        print(f"\tProcessed {i}/{len(texts)} ({i * 100 / len(texts):.2f}%)")

        batch_texts = texts[i:i + batch_size]
        docs.extend(list(nlp.pipe(batch_texts)))
    print("\tdone: 100.00%")

    # create DataFrame
    vectors = np.array([doc.vector for doc in docs])
    df = pd.DataFrame(vectors)
    df['PU'] = sentences['PU']

    # reorder columns if necessary
    df = df[['PU'] + [col for col in df.columns if col != 'PU']]

    # save the DataFrame
    utils.mkdir(utils.path_dirname(output_file))
    utils.csv_save(df, output_file)

    print("vectorization: Texts successfully vectorized and saved.")


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize CSV file")
    # args
    parser.add_argument('-i', '--csv-path', type=str, required=True,
                        help="Path(s) to the CSV file(s)")
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help="Output file to save the vectorize data")
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help="Batch size (int>0); def: 128")
    parser.add_argument('-m', '--mode', choices=['spacy', 'bow'],
                        default='spacy',
                        help="Vectorization using 'spacy' or 'bow'; def: spacy")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    if args.batch_size is not None:
        batch_size = args.batch_size
        if batch_size <= 0:
            print("vectorization: batch size must be a positive int")
            exit(1)

    if args.mode == 'spacy':
        vectorize_spacy(args.csv_path, args.output_file, batch_size)
    elif args.mode == 'bow':
        vectorize_bow(args.csv_path, args.output_file)

