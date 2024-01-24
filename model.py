"""
Word2Vec from Scratch: Models
"""
from dataclasses import dataclass
import torch.nn as nn

# constants to pre-process data and build vocab
MIN_WORD_FREQUENCY = 50
MAX_SEQUENCE_LENGTH = 256

# embedding constants
EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1  # keep embedding values low


class SkipGramModel(nn.Module):
    """
    Implementation of SkipGram model described in paper:
    https://arxiv.org/abs/1301.3781

    Takes a single word
    Returns logprobs for all possible neighbors

    inputs_ is a one_hot encoding of a word in vocabulary
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


class CBOWModel(SkipGramModel):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781

    Note that this class extends SkipGram because the
    network is identical with the exception that the input
    in this case is a group of words instead of one
    which is handled differently in the forward pass
    by averaging the embeddings for each input word

    Takes N*2 neighbors from each side
    Returns logprobs for entire vocabulary

    inputs_ is a one_hot encoding of BOWs (e.g. N_WORDS * 2) in vocabulary
    inputs_ is (N_WORDS * 2) by n; embeddings is n by m; linear is m by n
    input_ dot embeddings is (N_WORDS * 2) by m => x
    we then take the mean of x over the 1st axis (e.g the m'th axis or columns)
    then x becomes 1 by m
    and x dot linear is 1 by n
    """

    def forward(self, inputs_):
        """forward pass"""
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


@dataclass
class ModelConfig:
    """class to store Model configuration"""

    name: str
    model: nn.Module
    n_words: int


CBOW_CONFIG = ModelConfig(name="CBOWModel", model=CBOWModel, n_words=4)
SKIPGRAM_CONFIG = ModelConfig(name="SkipGramModel", model=SkipGramModel, n_words=4)
