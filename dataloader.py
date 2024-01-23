"""
Word2Vec from Scratch: Dataloader
Based on https://github.com/OlgaChernytska/word2vec-pytorch
Updated to avoid deprecated libraries like Torchtext
"""
from functools import partial
from collections import Counter
import re
from dataclasses import dataclass

# from line_profiler import profile
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# import nltk
# from nltk.tokenize import word_tokenize
from constants import (
    MIN_WORD_FREQUENCY,
    CBOW_N_WORDS,
    SKIPGRAM_N_WORDS,
    MAX_SEQUENCE_LENGTH,
)

# nltk.download("punkt")  # for tokenizer
# 12751034 re2 in function
# 376333 re in line
# 409510 re in function


def get_dataset(ds_name: str = "wikitext-2-v1"):
    """construct dataset"""
    if ds_name not in ["wikitext-103-v1", "wikitext-2-v1"]:
        raise ValueError("valid ds_name: 'wikitext-103-v1', 'wikitext-2-v1'")
    return load_dataset("wikitext", ds_name)


@dataclass
class Vocab:
    """class to store Vocabulary"""

    word_to_idx: dict
    idx_to_word: list


TOKENIZER = re.compile(r"\w+")


def tokenize(text):
    """basic tokenizer"""
    # the pervious solution was word_tokenizer from nltk
    return TOKENIZER.findall(text.lower())


def build_vocab(dataset):
    """Builds vocabulary from dataset"""
    word_freq = Counter()

    # for text in dataset["text"]:
    #     word_freq.update(tokenize(text))
    # optimized below by adding everything to counter in a big batch

    all_tokens = []
    for text in dataset["text"]:
        all_tokens.extend(tokenize(text))
    word_freq.update(all_tokens)

    vocab = {
        word: freq for word, freq in word_freq.items() if freq >= MIN_WORD_FREQUENCY
    }
    vocab["unk"] = 10**18  # very large number so it's idx=0

    word_to_idx = {
        word: idx
        for idx, (word, freq) in enumerate(
            # sort by the [1]st element in the tuple ('word', freq)
            sorted(vocab.items(), key=lambda item: item[1], reverse=True),
            # start = 1 # where to start counting for enumerate
        )
    }

    idx_to_word = [word for word, idx in word_to_idx.items()]

    return Vocab(word_to_idx, idx_to_word)


def collate_cbow(batch, word_to_idx):
    """
    Prepares examples for trainer
    `batch` is a list of text paragraphs

    Context for i-th word is CBOW_N_WORDS before and after

    Long paragraphs will be truncated to MAX_SEQUENCE_LENGTH

    Each element in `batch_input` is CBOW_N_WORDS * 2 words
    Each element in `batch_output` is the middel / i-th word
    """
    batch_input, batch_output = [], []
    for b in batch:
        text_tokens_ids = [word_to_idx.get(word, 0) for word in tokenize(b["text"])]

        # if the sentence is too short to extract CBOW, skip
        if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        # iterate for as long as the sliding window allows
        for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output_ = token_id_sequence.pop(CBOW_N_WORDS)  # get the middle word
            input_ = token_id_sequence  # the remaining words
            batch_input.append(input_)
            batch_output.append(output_)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def collate_skipgram(batch, word_to_idx):
    """
    Prepares examples for trainer
    `batch` is a list of text paragraphs

    Context for i-th word is SKIPGRAM_N_WORDS before and after

    Long paragraphs will be truncated to MAX_SEQUENCE_LENGTH

    Each element in `batch_input` is the middle / i-th word
    Each element in `batch_output` is one of SKIPGRAM_N_WORDS * 2 context words
    """
    batch_input, batch_output = [], []
    for b in batch:
        if isinstance(b["text"], str):
            text_tokens_ids = [word_to_idx.get(word, 0) for word in tokenize(b["text"])]

        # if the sentence is too short to extract CBOW, skip
        if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
            continue

        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

        # iterate for as long as the sliding window allows
        for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + SKIPGRAM_N_WORDS * 2 + 1)]
            input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)  # get the middle word
            outputs_ = token_id_sequence  # the remaining words
            for output_ in outputs_:
                batch_input.append(input_)
                batch_output.append(output_)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_and_vocab(
    model_class: str,
    batch_size: int,
    ds_name: str,
    ds_type: str = "train",
    shuffle: bool = True,
    vocab: Vocab = None,
):
    """Main interface for dataloader module"""

    if ds_type not in ["train", "validation", "test"]:
        raise ValueError("valid ds_type: 'train', 'validation', 'test'")

    dataset = get_dataset(ds_name)

    # word_to_idx, idx_to_word
    if vocab is None:
        vocab = build_vocab(dataset[ds_type])

    if model_class == "CBOWModel":
        collate_fn = partial(collate_cbow, word_to_idx=vocab.word_to_idx)
    elif model_class == "SkipGramModel":
        collate_fn = partial(collate_skipgram, word_to_idx=vocab.word_to_idx)
    else:
        raise ValueError("Model should be: cbow, skipgram")

    dataloader = DataLoader(
        dataset[ds_type], batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    return dataloader, vocab
