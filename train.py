"""
Word2Vec from Scratch: train / main
"""
import os
from dataloader import W2VDataLoader
from trainer import Trainer
from model import (
    CBOW_CONFIG,
    # SKIPGRAM_CONFIG,
)


def train():
    """main function to coordinate training"""
    model_config = CBOW_CONFIG  # CBOW_CONFIG , SKIPGRAM_CONFIG
    ds_name = "wikitext-2-v1"  # "wikitext-103-v1" (large) ,  "wikitext-2-v1" (small)
    model_dir = f"{model_config.name}_{ds_name}_data"
    batch_size = 96
    shuffle = True
    learning_rate = 0.025
    epochs = 5
    checkpoint_frequency = 5_000

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataloader = W2VDataLoader(
        batch_size=batch_size,
        ds_name=ds_name,
        shuffle=shuffle,
        model_config=model_config,
        model_dir=model_dir,
    )

    vocab_size = len(dataloader.vocab.word_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # initialize model instance
    model = model_config.model(vocab_size=vocab_size)

    trainer = Trainer(
        model=model,
        model_dir=model_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        dataloader=dataloader,
        checkpoint_frequency=checkpoint_frequency,
    )

    trainer.train()
    print("Training Finished.")

    trainer.save_model()
    trainer.save_loss()


if __name__ == "__main__":
    train()
