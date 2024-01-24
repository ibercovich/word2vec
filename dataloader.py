"""
Word2Vec from Scratch: Dataloader
"""
from collections import Counter
import pickle
import os
import re
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import ModelConfig, MIN_WORD_FREQUENCY, MAX_SEQUENCE_LENGTH


@dataclass
class Vocab:
    """class to store Vocabulary"""

    word_to_idx: dict
    idx_to_word: list


@dataclass
class W2VDataLoader:
    """builds dataloader and vocabulary"""

    model_config: ModelConfig
    batch_size: int
    ds_name: str
    shuffle: bool = True
    vocab: Vocab = None
    model_dir: str = None

    def __post_init__(self):
        valid_ds = ["wikitext-103-v1", "wikitext-2-v1"]
        if self.ds_name not in valid_ds:
            raise ValueError(f"valid ds_name: {valid_ds}")

        self.tokenizer = re.compile(r"\w+")

        self.dataset = load_dataset("wikitext", self.ds_name)
        if self.vocab is None:
            self.build_vocab()
            self.save_vocab()

        self.train = DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate,
        )

        self.val = DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate,
        )

    def tokenize(self, text):
        """basic tokenizer"""
        # the pervious solution was word_tokenizer from nltk
        return self.tokenizer.findall(text.lower())

    def build_vocab(self):
        """Builds vocabulary from dataset"""
        word_freq = Counter()

        all_tokens = []
        for text in self.dataset["train"]["text"]:
            # could add tokenize(text) directly to
            # word_freq.update() but it's a bit slower
            all_tokens.extend(self.tokenize(text))
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
            )
        }

        idx_to_word = [word for word, idx in word_to_idx.items()]
        self.vocab = Vocab(word_to_idx, idx_to_word)

    def save_vocab(self):
        """save vocabulary to disk"""
        if self.model_dir:
            # save vocabulary to disk
            vocab_path = os.path.join(self.model_dir, "vocab.pkl")
            with open(vocab_path, "wb") as file:
                pickle.dump(self.vocab, file)

    def collate(self, batch):
        """
        Prepares examples for trainer
        `batch` is a list of text paragraphs
        Long paragraphs will be truncated to MAX_SEQUENCE_LENGTH
        Context for i-th word is model_config.n_words before and after

        if model_config.name == "CBOWModel":
        Each element in `batch_input` is model_config.n_words * 2 words
        Each element in `batch_output` is the middle / i-th word

        elif model_config.name == "SkipGramModel":
        Each element in `batch_input` is the middle / i-th word
        Each element in `batch_output` is one of model_config.n_words* 2 context words

        """
        batch_input, batch_output = [], []
        for b in batch:
            text_tokens_ids = [
                self.vocab.word_to_idx.get(word, 0) for word in self.tokenize(b["text"])
            ]

            # if the sentence is too short to extract CBOW, skip
            if len(text_tokens_ids) < self.model_config.n_words * 2 + 1:
                continue

            if MAX_SEQUENCE_LENGTH:
                text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

            # iterate for as long as the sliding window allows
            for idx in range(len(text_tokens_ids) - self.model_config.n_words * 2):
                token_id_sequence = text_tokens_ids[
                    idx : (idx + self.model_config.n_words * 2 + 1)
                ]

                if self.model_config.name == "CBOWModel":
                    output_ = token_id_sequence.pop(
                        self.model_config.n_words
                    )  # get the middle word
                    inputs_ = token_id_sequence  # the remaining words
                    batch_input.append(inputs_)
                    batch_output.append(output_)

                elif self.model_config.name == "SkipGramModel":
                    input_ = token_id_sequence.pop(
                        self.model_config.n_words
                    )  # get the middle word
                    outputs_ = token_id_sequence  # the remaining words
                    # make one example per output
                    for output_ in outputs_:
                        batch_input.append(input_)
                        batch_output.append(output_)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output
