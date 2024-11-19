import argparse

import numpy as np
import spacy
import pandas as pd

import utils


# SpaCy #######################################################################
def gen_bow(path, vocab_file):
    """
    Generate Bag of Words (BoW) vocabulary and save it.
    :param path: Path to the CSV file
    :param vocab_file: File to save the vocabulary as a separate mapping
    """
    print("vectorization: Generating Bag of Words (BoW) vocabulary")
    # load
    sentences = utils.csv_load(path)
    texts = sentences['TXT'].values
    nlp = spacy.load("es_core_news_md")

    # gen
    print("vectorization:gen_bow: Tokenizing texts for vocabulary creation")
    vocab = set()
    for doc in nlp.pipe((text.lower() for text in texts)):
        vocab.update(t.text for t in doc if not (t.is_stop or t.is_punct))

    vocab = sorted(vocab)  # Sort for consistency
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    print(f"vectorization:gen_bow: vocabulary size {len(vocab)} words")

    # save
    vocab_df = pd.DataFrame({
        'index': list(word_to_idx.values()),
        'word': list(word_to_idx.keys())
    })
    utils.mkdir(utils.path_dirname(vocab_file))
    utils.csv_save(vocab_df, vocab_file)
    print("vectorization:gen_bow: BoW done")


def vectorize_bow(path, vocab_file, output_file):
    """
    Vectorize the input text using Bag of Words with index-based column names.
    :param path: Path to the CSV file
    :param vocab_file: File containing the vocabulary index mapping
    :param output_file: File to save the vectorized data
    """
    print("vectorization: Vectorizing using Bag of Words (BoW)")
    # load
    vocab_df = utils.csv_load(vocab_file)
    word_to_idx = {row['word']: row['index'] for _, row in vocab_df.iterrows()}
    sentences = utils.csv_load(path)
    texts = sentences['TXT'].values
    nlp = spacy.load("es_core_news_md")

    # tokenize and generate BoW vectors
    print("vectorization:vectorize_bow: Generating BoW vectors")
    bow_vectors = np.zeros((len(texts), len(word_to_idx)), dtype=int)
    for i, doc in enumerate(nlp.pipe((text.lower() for text in texts))):
        for token in doc:
            if token.text in word_to_idx:
                bow_vectors[i, word_to_idx[token.text]] += 1

    # create DataFrame
    bow_df = pd.DataFrame(bow_vectors)
    bow_df['PU'] = sentences['PU']

    # reorder columns if necessary
    bow_df = bow_df[['PU'] + [col for col in bow_df.columns if col != 'PU']]

    # save the DataFrame
    utils.mkdir(utils.path_dirname(output_file))
    utils.csv_save(bow_df, output_file)
    print("vectorization:vectorize_bow: Vectorization done")


# SpaCy #######################################################################
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
        print(f"\tProcessed {i:3d}/{len(texts)} ({i * 100 / len(texts):.2f}%)")

        batch_texts = texts[i:i + batch_size]
        docs.extend(list(nlp.pipe(batch_texts)))
    print(f"\tProcessed {len(texts)}/{len(texts)} (100.0%)")

    # create DataFrame
    vectors = np.array([doc.vector for doc in docs])
    df = pd.DataFrame(vectors)
    df['PU'] = sentences['PU']

    # reorder columns if necessary
    df = df[['PU'] + [col for col in df.columns if col != 'PU']]

    # save the DataFrame
    utils.mkdir(utils.path_dirname(output_file))
    utils.csv_save(df, output_file)

    print("vectorization:vectorize_spacy: Vectorization done")


def validate_args(args):
    """
    Validate command-line arguments.
    """
    if args.batch_size <= 0:
        print("vectorization: batch size must be a positive int")
        exit(1)

    if args.mode == 'bow' and not args.bow_file:
        print("vectorization: BoW file is required for 'bow' mode")
        exit(1)


###############################################################################
# main ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize CSV file")
    parser.add_argument('-i', '--csv-path', type=str, required=True,
                        help="Path to the CSV file")
    parser.add_argument('-o', '--output-file', type=str, required=True,
                        help="Output file to save the vectorize data or vocab")
    parser.add_argument('-m', '--mode', choices=['spacy', 'bow'],
                        default='spacy',
                        help="Vectorization using 'spacy' or 'bow'; def: spacy")
    parser.add_argument('-g', '--gen-bow', action='store_true',
                        help="Generate BoW vocabulary")
    parser.add_argument('-v', '--bow-file', type=str,
                        help="Path to the BoW file (required for 'bow' mode)")
    parser.add_argument('-b', '--batch-size', type=int, default=128,
                        help="Batch size (int>0); def: 128")

    try:
        args = parser.parse_args()
        validate_args(args)
    except SystemExit:
        exit(1)

    if args.gen_bow:
        gen_bow(args.csv_path, args.output_file)
    elif args.mode == 'bow':
        vectorize_bow(args.csv_path, args.bow_file, args.output_file)
    elif args.mode == 'spacy':
        vectorize_spacy(args.csv_path, args.output_file, args.batch_size)
    else:
        print("vectorization: Unknown mode")
        exit(1)

