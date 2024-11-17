from transformers import AutoTokenizer

import utils

# Load the tokenizer for distiluse-base-multilingual-cased-v2
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2')


def check_tokens(path):
    """
    Our model can handle a maximum of 512 tokens at a time. We will check if our texts comply with this.
    If a text exceeds this number of tokens, the model will not be able to process it properly.
    There are two ways to solve this problem: truncation and division.
          - Truncation: the tokens exceeding the limit will not be taken into account.
          - Division: we can take those large texts and divide them in order to get texts with fewer tokens.
    We will identify these texts and add them to a list, so that we can divide them later.
    :param path: The path of the file to load
    :return: bool, true if there are sentences that exceed the 512 limit
    """
    sentences = utils.csv_load(path)
    texts = sentences['TXT'].values

    # Store the texts that are longer than 512 tokens (max sequence length for the model)
    long_texts = []

    for i, text in enumerate(texts):
        tokenized = tokenizer(text, return_tensors='pt', truncation=False)
        num_tokens = len(tokenized['input_ids'][0])

        if num_tokens > 512:
            long_texts.append(i)

    return long_texts


def validate_tokens(text):
    """
    Check if the given text fits our tokenization.
    :param text: The text to validate
    :return: bool, number of tokens is valid
    """
    tokenized = tokenizer(text, return_tensors='pt', truncation=False)
    num_tokens = len(tokenized['input_ids'][0])
    return num_tokens <= 512


def tokenize(texts):
    """
    Tokenize the given text.
    :param texts: The text to tokenize
    :return: The input IDs and the attention mask
    """
    tokenized = tokenizer(texts, return_tensors='pt', padding=True, truncation=False)
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    return input_ids, attention_mask
