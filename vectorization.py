import argparse
import gc

import numpy as np
import pandas as pd
import torch

# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer

import tokenization
import utils
import preprocess


def vectorize(path, output_dir, batch_size=128):
    """
    Vectorize the input text.
    :param path: The text to vectorize
    :param batch_size: The size of the batches
    """
    print(f"vectorization:vectorize: checking tokens", flush=True)
    if tokenization.check_tokens(path):
        directory = utils.path_dirname(path)
        conf_file = utils.path_join(directory, 'preprocess_conf.json')
        if utils.path_is_file(conf_file):
            preprocess.data_reparse(conf_file)
        else:
            print(f"vectorization:vectorize: the configuration file doesn't exist")
            print(f"\ttext might be truncated")

    gc.collect()
    print("vectorization:vectorize: vectorizing", flush=True)
    sentences = utils.csv_load(path)
    texts = sentences['TXT'].values
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    """
    We can use the following line to get the embeddings. 
    embeddings = model.encode(texts)
    This handles the entire pipeline: tokenization, padding, truncation and generating the embeddings.

    However, we can follow this steps manually and specify options.
    First, we will tokenize the texts and get the input_ids and attention_masks.
      - input_ids: tokenized version of the text. Token IDs.
      - attention_mask: helps distinguish padding tokens.
    Then, we can calculate the embeddings using these.
    """

    all_embeddings = []

    torch.manual_seed(42)  # for reproducibility reasons
    np.random.seed(42)
    
    print("vectorization:vectorize: batching", flush=True)
    for i in range(0, len(texts), batch_size):
        print("\tdone: {0:.2f}%".format(i*100/len(texts)))
        batch_texts = texts[i:i + batch_size]

        input_ids, attention_mask = tokenization.tokenize(batch_texts.tolist())

        with torch.no_grad():  # Avoid gradient calculations
            batch_embeddings = model({'input_ids': input_ids, 'attention_mask': attention_mask})

        """
        batch_embeddings is a dictionary with the following fields:
        input_ids, attention_mask, token_embeddings, all_layer_embeddings, sentence_embedding
         - token_embeddings: raw embeddings for each token. [batch_size, sequence_length, embedding_dim]
         - all_layer_embeddings: for each transformer block. [batch_size, num_layers, sequence_length, embedding_dim]
         - sentence_embedding: final sentence embeddings. [batch_size, embedding_dim]
        sentence_embeddings is the result we want, which is of type torch.Tensor.
        We'll convert it to a numpy array (numpy.ndarray) to be able to save it to a .csv.
        """

        batch_sentence_embedding = batch_embeddings['sentence_embedding'].cpu().numpy()
        all_embeddings.append(batch_sentence_embedding)
    print(f"\tdone: 100.00%")

    print("vectorization:vectorize: merging batches", flush=True)
    sentence_embedding = np.vstack(all_embeddings)

    df = pd.DataFrame(sentence_embedding)
    df['PU'] = sentences['PU']
    df = df[['PU'] + [col for col in df.columns if col != 'PU']]

    basename = utils.path_basename(path)
    filename = utils.path_join(output_dir, basename)
    utils.mkdir(output_dir)
    utils.csv_save(df, filename)

    print(f"vectorization:vectorize: Texts successfully vectorized and saved.")


###############################################################################
## main #######################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize CSV file(s)")
    # args
    parser.add_argument('csv_path', type=str, nargs='+',
                        help="Path(s) to the CSV file(s)")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Output directory to save the vectorize data")
    parser.add_argument('-b', '--batch-size', type=int,
                        help="Batch size (int>0); def: 128")

    try:
        args = parser.parse_args()
    except SystemExit:
        exit(1)

    if args.batch_size is not None:
        batch_size = args.batch_size
        if batch_size <= 0:
            print("vectorization: batch size must be a positive int")
            exit(1)
    else:
        batch_size = 128

    for path in args.csv_path:
        vectorize(path, args.output_dir, batch_size)
