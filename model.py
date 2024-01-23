"""
Word2Vec from Scratch: Models
Based on https://github.com/OlgaChernytska/word2vec-pytorch
Updated to avoid deprecated libraries like Torchtext
"""
import torch.nn as nn
from constants import EMBED_DIMENSION, EMBED_MAX_NORM


class CBOWModel(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781

    Takes N*2 neighbors from each side
    Returns logprobs for entire vocabulary

    inputs_ is a one_hot encoding of BOWs (e.g. N_WORDS * 2) in vocabulary
    inputs_ is (N_WORDS * 2) by n; embeddings is n by m; linear is m by n
    input_ dot embeddings is (N_WORDS * 2) by m => x
    we then take the mean of x over the 1st axis (e.g the m'th axis or columns)
    then x becomes 1 by m
    and x dot linear is 1 by n
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # n by m Matrix
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,  # n
            embedding_dim=EMBED_DIMENSION,  # m
            max_norm=EMBED_MAX_NORM,
        )
        # m by n Matrix
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,  # m
            out_features=vocab_size,  # n
        )

    def forward(self, inputs_):
        """forward pass"""
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGramModel(nn.Module):
    """
    Implementation of SkipGram model described in paper:
    https://arxiv.org/abs/1301.3781

    Takes a single word
    Returns logprobs for all possible neighbors

    inputs_ is a one_hot encoding of word in vocabulary
    inputs_ is 1 by n; embeddings is n by m; linear is m by n
    input_ dot embeddings is 1 by m => x
    and x dot linear is 1 by n
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        # n by m Matrix
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,  # n
            embedding_dim=EMBED_DIMENSION,  # m
            max_norm=EMBED_MAX_NORM,
        )
        # m by n Matrix
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,  # m
            out_features=vocab_size,  # n
        )

    def forward(self, inputs_):
        """forward pass"""
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x
