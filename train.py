"""
Word2Vec from Scratch: train
Based on https://github.com/OlgaChernytska/word2vec-pytorch
Updated to avoid deprecated libraries like Torchtext
"""
import os
import pickle
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from dataloader import get_dataloader_and_vocab
from model import CBOWModel, SkipGramModel
from trainer import Trainer


def train():
    """main function to coordinate training"""
    model_class = "CBOWModel"  # "SkipGramModel"  "CBOWModel"
    ds_name = "wikitext-2-v1"  # "wikitext-103-v1"   "wikitext-2-v1"
    model_dir = f"{model_class}_{ds_name}_data"
    batch_size = 96
    shuffle = True
    learning_rate = 0.025
    epochs = 5
    checkpoint_frequency = 5_000

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_dataloader, vocab = get_dataloader_and_vocab(
        model_class=model_class,
        ds_name=ds_name,
        batch_size=batch_size,
        ds_type="train", # "train"
        shuffle=shuffle,
    )

    val_dataloader, _ = get_dataloader_and_vocab(
        model_class=model_class,
        ds_name=ds_name,
        batch_size=batch_size,
        ds_type="validation",
        shuffle=shuffle,
        vocab=vocab,
    )

    vocab_size = len(vocab.word_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    if model_class == "CBOWModel":
        model = CBOWModel(vocab_size=vocab_size)
    elif model_class == "SkipGramModel":
        model = SkipGramModel(vocab_size=vocab_size)
    else:
        raise ValueError("valid model_class: 'CBOWModel', 'SkipGramModel'")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = LambdaLR(
        optimizer,
        # variable learning rate that ends at 0 in the last epoch
        lr_lambda=lambda epoch: (epochs - epoch) / epochs,
        verbose=True,
    )

    trainer = Trainer(
        model=model,
        model_dir=model_dir,
        epochs=epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=checkpoint_frequency,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()
    print("Training Finished.")

    trainer.save_model()
    trainer.save_loss()

    vocab_path = os.path.join(model_dir, "vocab.pkl")
    with open(vocab_path, "wb") as file:
        pickle.dump(vocab, file)

    print("Model artifacts saved to folder: ", model_dir)

if __name__ == '__main__':
    train()
    